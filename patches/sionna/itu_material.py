#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""ITU radio materials"""

import mitsuba as mi
from typing import Tuple, Callable

from .itu import itu_material, ITU_MATERIALS_PROPERTIES
from .radio_material import RadioMaterial


class ITURadioMaterial(RadioMaterial):
    # pylint: disable=line-too-long
    r"""
    Class implementing the materials defined in the ITU-R P.2040-3 recommendation [ITU_R_2040_3]_
    """

    ITU_MATERIAL_COLORS = {
        "marble": (0.701, 0.644, 0.485),
        "concrete": (0.539, 0.539, 0.539),
        "wood": (0.266, 0.109, 0.060),
        "metal": (0.220, 0.220, 0.254),
        "brick": (0.402, 0.112, 0.087),
        "glass": (0.168, 0.139, 0.509),
        "floorboard": (0.539, 0.386, 0.025),
        "ceiling_board": (0.376, 0.539, 0.117),
        "chipboard": (0.509, 0.159, 0.323),
        "plasterboard": (0.051, 0.539, 0.133),
        "plywood": (0.136, 0.076, 0.539),
        "very_dry_ground": (0.539, 0.319, 0.223),
        "medium_dry_ground": (0.539, 0.181, 0.076),
        "wet_ground": (0.539, 0.027, 0.147)
    }

    def __init__(
        self,
        name: str | None = None,
        itu_type: str | None = None,
        thickness: float | mi.Float | None = None,
        scattering_coefficient: float | mi.Float = 0.0,
        xpd_coefficient: float | mi.Float = 0.0,
        scattering_pattern: Callable[[mi.Vector3f, mi.Vector3f, ...], mi.Float] | None = None,
        color: Tuple[float, float, float] | None = None,
        props: mi.Properties | None = None,
        **kwargs):

        has_props = props is not None
        if has_props:
            direct_args_none = (
                (name is None) and (itu_type is None) and (thickness is None)
                and (scattering_coefficient == 0.0) and (xpd_coefficient == 0.0)
            )
            if not direct_args_none:
                raise ValueError(
                    "When providing a `props` dictionary, no argument other "
                    "than `scattering_pattern` and `color` should be provided."
                )
            if 'type' not in props:
                raise ValueError(
                    'Missing property "type" (string) to select the ITU material type.'
                )
            itu_type = props['type']
            del props['type']

        if itu_type not in ITU_MATERIALS_PROPERTIES:
            raise ValueError(f'Invalid ITU material type "{itu_type}"')
        self._itu_type = itu_type

        if color is None:
            color = ITURadioMaterial.ITU_MATERIAL_COLORS[itu_type]
            if has_props:
                props["color"] = mi.ScalarColor3f(color)

        # IMPORTANT FIX: never let scattering_pattern stay None
        if scattering_pattern is None:
            scattering_pattern = "lambertian"

        # Frequency update callback
        def cb(f: float):
            return itu_material(itu_type, f)

        if has_props:
            super().__init__(scattering_pattern=scattering_pattern,
                             frequency_update_callback=cb,
                             props=props,
                             **kwargs)
        else:
            super().__init__(name=name,
                             thickness=thickness,
                             scattering_coefficient=scattering_coefficient,
                             xpd_coefficient=xpd_coefficient,
                             scattering_pattern=scattering_pattern,
                             frequency_update_callback=cb,
                             color=color,
                             **kwargs)

    @property
    def itu_type(self):
        return self._itu_type

    def to_string(self) -> str:
        s = f"ITURadioMaterial type={self._itu_type}\n"\
            f"                 eta_r={self._eta_r[0]:.3f}\n"\
            f"                 sigma={self._sigma[0]:.3f}\n"\
            f"                 thickness={self._d[0]:.3f}\n"\
            f"                 scattering_coefficient={self._s[0]:.3f}\n"\
            f"                 xpd_coefficient={self._kx[0]:.3f}"
        return s


mi.register_bsdf("itu-radio-material",
                 lambda props: ITURadioMaterial(props=props))
