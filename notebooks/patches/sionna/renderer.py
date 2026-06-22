#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Rendering module of Sionna RT"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
import numpy as np

from sionna import rt
from .utils import (
    make_render_sensor,
    paths_to_segments,
    unmultiply_alpha,
    twosided_diffuse,
    radio_map_to_emissive_shape,
    scoped_set_log_level,
    scene_scale,
    clone_mesh,
    rotation_matrix,
)


def render(scene: rt.Scene,
           camera: str | rt.Camera | mi.ScalarTransform4f | mi.Sensor,
           paths: rt.Paths | None,
           show_devices: bool,
           show_orientations: bool,
           num_samples: int,
           resolution: tuple[int, int],
           fov: float | None = None,
           clip_at: float | None = None,
           clip_plane_orientation: tuple[float, float, float] = (0, 0, -1),
           max_depth: int = 8,
           radio_map: rt.RadioMap | None = None,
           rm_tx: int | str | None = None,
           rm_db_scale: bool = True,
           rm_cmap: str | callable | None = None,
           rm_vmin: float | None = None,
           rm_vmax: float | None = None,
           rm_metric: str = "path_gain",
           envmap: str | None = None,
           lighting_scale: float = 1.0) -> mi.Bitmap:
    r"""
    Renders two images with path tracing:
    1. Base scene with the meshes
    2. Paths, radio devices and radio map,
    then composites them together.
    """
    rendering_variant = (
        "cuda_ad_rgb"
        if dr.backend_v(mi.Float) == dr.JitBackend.CUDA
        else "llvm_ad_rgb"
    )

    with mi.util.scoped_set_variant(rendering_variant, "cuda_ad_rgb", "llvm_ad_rgb"):
        sensor = make_render_sensor(scene, camera=camera, resolution=resolution, fov=fov)
        exclude_mesh_ids = set()

        rm_is_part_of_scene = False
        if isinstance(radio_map, rt.MeshRadioMap) and not rm_is_part_of_scene:
            exclude_mesh_ids.add(radio_map.measurement_surface.id())

        visual_scene = visual_scene_from_wireless_scene(
            scene,
            sensor=sensor,
            max_depth=max_depth,
            clip_at=clip_at,
            clip_plane_orientation=clip_plane_orientation,
            envmap=envmap,
            lighting_scale=lighting_scale,
            exclude_mesh_ids=exclude_mesh_ids,
        )
        visual_scene = mi.load_dict(visual_scene)

        # Render the scene geometry
        main_image = mi.render(visual_scene, spp=num_samples).numpy()

        # Overlay scene
        overlay_scene = get_overlay_scene(
            scene,
            sensor,
            paths=paths,
            show_sources=show_devices,
            show_targets=show_devices,
            show_orientations=show_orientations,
            radio_map=radio_map,
            rm_tx=rm_tx,
            rm_db_scale=rm_db_scale,
            rm_cmap=rm_cmap,
            rm_vmin=rm_vmin,
            rm_vmax=rm_vmax,
            rm_metric=rm_metric,
        )

        if not overlay_scene:
            return main_image

        with scoped_set_log_level(mi.LogLevel.Error):
            overlay_scene = mi.load_dict(overlay_scene)

        depth_integrator = mi.load_dict({"type": "depth"})
        clipped_depth_integrator = depth_integrator
        if clip_at is not None:
            clipped_depth_integrator = visual_scene.integrator().as_depth_integrator()

        unclipped_integrator = mi.load_dict({
            "type": "path",
            "max_depth": max_depth,
            "hide_emitters": False,
        })

        depth1 = mi.render(
            visual_scene,
            sensor=sensor,
            integrator=clipped_depth_integrator,
            spp=4
        )
        depth1 = unmultiply_alpha(depth1.numpy())

        overlay_image = mi.render(
            overlay_scene,
            sensor=sensor,
            integrator=unclipped_integrator,
            spp=num_samples
        )
        overlay_image = overlay_image.numpy()

        depth2 = mi.render(
            overlay_scene,
            sensor=sensor,
            integrator=depth_integrator,
            spp=num_samples
        )
        depth2 = unmultiply_alpha(depth2.numpy())

        alpha1 = main_image[:, :, 3]
        alpha2 = overlay_image[:, :, 3]
        composite = overlay_image + main_image * (1 - alpha2[:, :, None])

        prefer_overlay = (alpha1[:, :, None] < 0.1) & (depth1 < 2 * depth2)
        if rm_is_part_of_scene:
            prefer_overlay |= np.abs(depth1 - depth2) < 0.01 * np.abs(depth1)

        result = np.where(
            (alpha1[:, :, None] > 0) & (depth1 < depth2) & (~prefer_overlay),
            main_image,
            composite
        )
        result[:, :, 3] = np.maximum(main_image[:, :, 3], composite[:, :, 3])

        return mi.Bitmap(result)


def visual_scene_from_wireless_scene(scene: rt.Scene,
                                     sensor: mi.Sensor,
                                     max_depth: int = 8,
                                     clip_at: float | None = None,
                                     clip_plane_orientation: tuple[float, float, float] = (0, 0, -1),
                                     envmap: str | None = None,
                                     lighting_scale: float = 1.0,
                                     exclude_mesh_ids: set[str] | None = None) -> dict:
    if dr.size_v(mi.Spectrum) != 3:
        raise ValueError(
            "This function is expected to be run using a rendering-focused "
            f"Mitsuba variant such as 'cuda_ad_rgb', but found '{mi.variant()}'."
        )

    bbox: mi.ScalarBoundingBox3f = scene.mi_scene.bbox()

    result = {
        "type": "scene",
        "sensor": sensor,
    }

    integrator = {
        "type": "sliced_path",
        "hide_emitters": True,
        "max_depth": max_depth,
    }

    if clip_at is not None:
        integrator["type"] = "sliced_path"
        integrator["slice1"] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f()
            .look_at(
                origin=(0, 0, 0),
                target=-mi.ScalarPoint3f(clip_plane_orientation),
                up=(0, 1, 0),
            )
            .translate([0, 0, clip_at])
            .scale(1.2 * bbox.extents())
        }
    else:
        integrator["type"] = "path"

    result["integrator"] = integrator

    if envmap:
        emitter = {
            "type": "envmap",
            "filename": envmap,
            "scale": lighting_scale,
            "to_world": mi.ScalarTransform4f()
            .rotate(axis=[1, 0, 0], angle=90)
            .rotate(axis=[0, 1, 0], angle=45),
        }
    else:
        emitter = {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": [lighting_scale, lighting_scale, lighting_scale],
            },
        }

    result["emitter"] = emitter

    bsdfs = {}
    for name, mat in scene.radio_materials.items():
        bsdfs[name] = twosided_diffuse(mat.color)

    default_bsdf = twosided_diffuse((0.7, 0.7, 0.7))

    for i, sh in enumerate(scene.mi_scene.shapes()):
        assert sh.is_mesh()
        if exclude_mesh_ids and sh.id() in exclude_mesh_ids:
            continue

        original_id = sh.bsdf().name
        new_id = f"shape-{i}-{sh.id()}"
        props = mi.Properties()
        props["bsdf"] = bsdfs.get(original_id, default_bsdf)
        result[new_id] = clone_mesh(sh, name=new_id, props=props)

    return result


def get_overlay_scene(scene: rt.Scene,
                      sensor: mi.Sensor,
                      paths: any | None = None,
                      show_sources: bool = True,
                      show_targets: bool = True,
                      show_orientations: bool = False,
                      radio_map: rt.RadioMap | None = None,
                      rm_tx: int | str | None = None,
                      rm_db_scale: bool = True,
                      rm_cmap: str | callable | None = None,
                      rm_vmin: float | None = None,
                      rm_vmax: float | None = None,
                      rm_metric: str = "path_gain") -> dict:
    result = {
        "type": "scene",
        "sensor": sensor,
    }

    sc = scene_scale(scene)
    if sc == 0.0:
        sc = 1.0
    radius = max(0.005 * sc, 0.5)

    for prefix, devices, enabled in (
        ("source", scene.transmitters, show_sources),
        ("target", scene.receivers, show_targets),
    ):
        if not enabled:
            continue

        for name, rd in devices.items():
            key = f"rd-{prefix}-{name}"
            if rd.display_radius is not None:
                display_radius = rd.display_radius
            else:
                display_radius = radius

            rd_pos = rd.position.numpy().squeeze()
            result[key] = {
                "type": "sphere",
                "center": rd_pos,
                "radius": display_radius,
                "light": {
                    "type": "area",
                    "radiance": {"type": "rgb", "value": rd.color},
                },
            }

            if show_orientations:
                line_length = 3 * display_radius
                n = rotation_matrix(rd.orientation) @ mi.ScalarNormal3f(1, 0, 0)
                n = n.numpy().squeeze()
                n_norm = np.linalg.norm(n)
                if n_norm > 0:
                    result[key + "-orientation"] = {
                        "type": "cylinder",
                        "p0": rd_pos,
                        "p1": rd_pos + line_length * (n / n_norm),
                        "radius": 0.1 * display_radius,
                        "light": {
                            "type": "area",
                            "radiance": {"type": "rgb", "value": rd.color},
                        },
                    }

    # Safe path overlay block
    if paths is not None:
        segs = paths_to_segments(paths)
        if segs is not None:
            starts, ends, colors = segs
            radius = min(0.20, 0.005 * sc)

            for i, (s, e, c) in enumerate(zip(starts, ends, colors)):
                result[f"path-{i:06d}"] = {
                    "type": "cylinder",
                    "p0": s,
                    "p1": e,
                    "radius": radius,
                    "light": {
                        "type": "area",
                        "radiance": {"type": "rgb", "value": c},
                    },
                }

    if radio_map is not None:
        result["radio-map"] = radio_map_to_emissive_shape(
            radio_map,
            tx=rm_tx,
            db_scale=rm_db_scale,
            rm_cmap=rm_cmap,
            vmin=rm_vmin,
            vmax=rm_vmax,
            rm_metric=rm_metric,
            viewpoint=sensor.world_transform().translation(),
        )

    return result