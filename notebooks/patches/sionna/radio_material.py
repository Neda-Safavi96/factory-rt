#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class implementing a radio material"""

import drjit as dr
import mitsuba as mi
from typing import Tuple, Callable

from sionna.rt.utils import (
    itu_coefficients_single_layer_slab,
    complex_relative_permittivity,
    jones_matrix_to_world_implicit,
    f_utd,
    jones_matrix_rotator,
    implicit_basis_vector,
    wedge_interior_angle,
    cot,
    sample_keller_cone,
)
from sionna.rt.constants import (
    InteractionType,
    DEFAULT_THICKNESS,
    DEFAULT_FREQUENCY,
    NO_JONES_MATRIX,
)
from .radio_material_base import RadioMaterialBase
from .scattering_pattern import scattering_pattern_registry, ScatteringPattern

from scipy.constants import speed_of_light


class RadioMaterial(RadioMaterialBase):
    # pylint: disable=line-too-long
    r"""
    Class implementing the radio material model described in the Primer on Electromagnetics.
    """

    def __init__(
        self,
        name: str | None = None,
        thickness: float | mi.Float = DEFAULT_THICKNESS,
        relative_permittivity: float | mi.Float = 1.0,
        conductivity: float | mi.Float = 0.0,
        scattering_coefficient: float | mi.Float = 0.0,
        xpd_coefficient: float | mi.Float = 0.0,
        scattering_pattern: str = "lambertian",
        frequency_update_callback: Callable[[mi.Float], Tuple[mi.Float, mi.Float]] | None = None,
        color: Tuple[float, float, float] | None = None,
        props: mi.Properties | None = None,
        **kwargs,
    ):

        if props is None:
            props = self._build_mi_props_from_params(
                name,
                thickness,
                relative_permittivity,
                conductivity,
                scattering_coefficient,
                xpd_coefficient,
                color,
                **kwargs,
            )

        # Real part of the relative permittivity
        eta_r = 1.0
        if "relative_permittivity" in props:
            eta_r = props["relative_permittivity"]
            del props["relative_permittivity"]
        self.relative_permittivity = eta_r

        # Conductivity [S/m]
        sigma = 0.0
        if "conductivity" in props:
            sigma = props["conductivity"]
            del props["conductivity"]
        self.conductivity = sigma

        # Material thickness [m]
        if "thickness" in props:
            d = props["thickness"]
            del props["thickness"]
        else:
            d = DEFAULT_THICKNESS
        self.thickness = d

        # Scattering coefficient
        s = 0.0
        if "scattering_coefficient" in props:
            s = props["scattering_coefficient"]
            del props["scattering_coefficient"]
        self.scattering_coefficient = s

        # XPD coefficient
        kx = 0.0
        if "xpd_coefficient" in props:
            kx = props["xpd_coefficient"]
            del props["xpd_coefficient"]
        self.xpd_coefficient = kx

        super().__init__(props)

        # Gather remaining properties as scattering pattern kwargs
        scattering_pattern_attributes = {}
        for prop_name in props.keys():
            scattering_pattern_attributes[prop_name] = props[prop_name]

        if scattering_pattern is None:
            scattering_pattern = "lambertian"
        factory = scattering_pattern_registry.get(scattering_pattern)
        self.scattering_pattern = factory(**scattering_pattern_attributes)

        self.frequency_update_callback = frequency_update_callback

    @RadioMaterialBase.scene.setter
    def scene(self, scene):
        RadioMaterialBase.scene.fset(self, scene)
        self.frequency_update()

    @property
    def relative_permittivity(self):
        return self._eta_r

    @relative_permittivity.setter
    def relative_permittivity(self, eta_r):
        if eta_r < 1.0:
            raise ValueError("Real part of the relative permittivity must be greater or equal to 1")
        self._eta_r = mi.Float(eta_r)

    @property
    def conductivity(self):
        return self._sigma

    @conductivity.setter
    def conductivity(self, sigma):
        if sigma < 0.0:
            raise ValueError("The conductivity must be greater or equal to 0")
        self._sigma = mi.Float(sigma)

    @property
    def thickness(self):
        return self._d

    @thickness.setter
    def thickness(self, d):
        if d < 0.0:
            raise ValueError("The material thickness must be positive")
        self._d = mi.Float(d)

    @property
    def scattering_coefficient(self):
        return self._s

    @scattering_coefficient.setter
    def scattering_coefficient(self, s):
        if s < 0.0 or s > 1.0:
            raise ValueError("Scattering coefficient must be in range (0,1)")
        self._s = mi.Float(s)

    @property
    def xpd_coefficient(self):
        return self._kx

    @xpd_coefficient.setter
    def xpd_coefficient(self, kx):
        if kx < 0.0 or kx > 1.0:
            raise ValueError("XPD coefficient must be in the range (0,1)")
        self._kx = mi.Float(kx)
        self._build_xpd_jones_mat()

    @property
    def scattering_pattern(self):
        return self._scattering_pattern

    @scattering_pattern.setter
    def scattering_pattern(self, sp):
        if not isinstance(sp, ScatteringPattern):
            raise ValueError("Not an instance of ScatteringPattern")
        self._scattering_pattern = sp

    @property
    def frequency_update_callback(self):
        return self._frequency_update_callback

    @frequency_update_callback.setter
    def frequency_update_callback(self, value):
        self._frequency_update_callback = value
        self.frequency_update()

    def frequency_update(self):
        if self._frequency_update_callback is None:
            return
        if self.scene is None:
            return

        relative_permittivity, conductivity = self._frequency_update_callback(self.scene.frequency)
        self.relative_permittivity = relative_permittivity
        self.conductivity = conductivity

    def sample(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: mi.Float,
        sample2: mi.Point2f,
        active: bool | mi.Bool = True,
    ) -> Tuple[mi.BSDFSample3f, mi.Spectrum]:
        # Incident direction of propagation in the local frame
        ki_local = si.wi
        cos_theta_i = -ki_local.z

        compute_jones_matrix = (ctx.component & NO_JONES_MATRIX) == 0
        diffraction_enabled = (ctx.component & InteractionType.DIFFRACTION) > 0

        # Local enabled interaction flags
        loc_en_inter = dr.reinterpret_array(mi.UInt, si.dp_du.z)
        diffraction = active & (loc_en_inter == InteractionType.DIFFRACTION)

        if self.scene is None:
            angular_frequency = dr.two_pi * DEFAULT_FREQUENCY
            wavelength = speed_of_light / DEFAULT_FREQUENCY
            wavenumber = dr.two_pi / wavelength
        else:
            angular_frequency = self.scene.angular_frequency
            wavelength = self.scene.wavelength
            wavenumber = self.scene.wavenumber

        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T

        eta = complex_relative_permittivity(self._eta_r, self._sigma, angular_frequency)

        r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(
            cos_theta_i, eta, self._d, wavelength
        )

        sampled_event, probs, specular, diffuse, _ = self._sample_event_type(
            sample1, r_te, r_tm, t_te, t_tm, loc_en_inter
        )
        reflection = specular | diffuse
        sampled_event[diffraction] = InteractionType.DIFFRACTION

        ko_spec_trans_local = self._specular_reflection_transmission_direction(
            ki_local, reflection
        )
        ko_diffuse_local = self._diffuse_reflection_direction(sample2)

        if diffraction_enabled:
            ko_diffr_local = self._diffraction_direction(si, sample2.x, ki_local)
        else:
            ko_diffr_local = mi.Vector3f(0.0)

        if compute_jones_matrix:
            spec_trans_mat = self._specular_reflection_transmission_matrix(
                to_world, ki_local, ko_spec_trans_local, reflection, r_te, r_tm, t_te, t_tm
            )

            diff_mat = self._diffuse_reflection_matrix(
                si, ki_local, ko_diffuse_local, spec_trans_mat
            )

            if diffraction_enabled:
                diffr_mat = self._diffraction_matrix(
                    to_world, ki_local, ko_diffr_local, si, eta, wavenumber
                )
            else:
                diffr_mat = mi.Matrix4f(0.0)

            jones_mat = dr.select(diffuse, diff_mat, spec_trans_mat)
            jones_mat = dr.select(diffraction, diffr_mat, jones_mat)

            s = dr.select(reflection, dr.sqrt(1.0 - dr.square(self._s)), 1.0)
            s = dr.select(diffuse, self._s, s)
            jones_mat *= s * dr.detach(dr.rsqrt(probs))

            # Keep Mitsuba/Sionna happy without the old broken fallback hacks
            try:
                jones_mat = mi.Spectrum(jones_mat)
            except Exception:
                jones_mat = mi.Spectrum(1.0)
        else:
            jones_mat = mi.Spectrum(0.0)

        ko_local = dr.select(diffuse, ko_diffuse_local, ko_spec_trans_local)
        ko_local = dr.select(diffraction, ko_diffr_local, ko_local)

        bs = mi.BSDFSample3f()
        bs.sampled_component = sampled_event
        bs.wo = si.to_world(ko_local)
        bs.pdf = probs
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DeltaReflection)
        bs.eta = 1.0

        return bs, jones_mat

    def eval(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True,
    ) -> mi.Spectrum:
        ki_local = si.wi
        cos_theta_i = -ki_local.z

        diffraction_enabled = (ctx.component & InteractionType.DIFFRACTION) > 0

        if self.scene is None:
            angular_frequency = dr.two_pi * DEFAULT_FREQUENCY
            wavelength = speed_of_light / DEFAULT_FREQUENCY
            wavenumber = dr.two_pi / wavelength
        else:
            angular_frequency = self.scene.angular_frequency
            wavelength = self.scene.wavelength
            wavenumber = self.scene.wavenumber

        ko_world = wo
        ko_local = si.to_local(ko_world)

        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T

        eta = complex_relative_permittivity(self._eta_r, self._sigma, angular_frequency)

        r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(
            cos_theta_i, eta, self._d, wavelength
        )

        sampled_event = dr.reinterpret_array(mi.UInt, si.dp_du.z)
        reflection = (sampled_event == InteractionType.SPECULAR) | (
            sampled_event == InteractionType.DIFFUSE
        )
        diffuse = sampled_event == InteractionType.DIFFUSE
        diffraction = sampled_event == InteractionType.DIFFRACTION

        ko_spec_trans_local = self._specular_reflection_transmission_direction(
            ki_local, reflection
        )

        spec_trans_mat = self._specular_reflection_transmission_matrix(
            to_world, ki_local, ko_spec_trans_local, reflection, r_te, r_tm, t_te, t_tm
        )

        diff_mat = self._diffuse_reflection_matrix(si, ki_local, ko_local, spec_trans_mat)

        if diffraction_enabled:
            diffr_mat = self._diffraction_matrix(
                to_world, ki_local, ko_local, si, eta, wavenumber
            )
        else:
            diffr_mat = mi.Matrix4f(0.0)

        jones_mat = dr.select(diffuse, diff_mat, spec_trans_mat)
        jones_mat = dr.select(diffraction, diffr_mat, jones_mat)

        s = dr.select(reflection, dr.sqrt(1.0 - dr.square(self._s)), 1.0)
        s = dr.select(diffuse, self._s, s)
        jones_mat *= s

        try:
            jones_mat = mi.Spectrum(jones_mat)
        except Exception:
            jones_mat = mi.Spectrum(1.0)

        return jones_mat

    def pdf(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True,
    ) -> mi.Float:
        ki_local = si.wi
        cos_theta_i = -ki_local.z

        if self.scene is None:
            angular_frequency = dr.two_pi * DEFAULT_FREQUENCY
            wavelength = speed_of_light / DEFAULT_FREQUENCY
        else:
            angular_frequency = self.scene.angular_frequency
            wavelength = self.scene.wavelength

        loc_en_inter = dr.reinterpret_array(mi.UInt, si.dp_du.z)

        eta = complex_relative_permittivity(self._eta_r, self._sigma, angular_frequency)

        r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(
            cos_theta_i, eta, self._d, wavelength
        )

        sampled_event = dr.reinterpret_array(mi.UInt, si.dp_du.z)
        specular = sampled_event == InteractionType.SPECULAR
        transmission = sampled_event == InteractionType.REFRACTION
        diffraction = sampled_event == InteractionType.DIFFRACTION

        prs, prd, pt, pd = self._event_probabilities(r_te, r_tm, t_te, t_tm, loc_en_inter)

        probs = dr.select(specular, prs, prd)
        probs[transmission] = pt
        probs[diffraction] = pd
        return probs

    def traverse(self, callback: mi.TraversalCallback):
        callback.put("eta_r", self._eta_r, mi.ParamFlags.Differentiable)
        callback.put("sigma", self._sigma, mi.ParamFlags.Differentiable)
        callback.put("d", self._d, mi.ParamFlags.Differentiable)
        callback.put("s", self._s, mi.ParamFlags.Differentiable)
        callback.put("xpd_coefficient", self._kx, mi.ParamFlags.Differentiable)

    def to_string(self) -> str:
        s = (
            f"RadioMaterial eta_r={self._eta_r[0]:.3f}\n"
            f"              sigma={self._sigma[0]:.3f}\n"
            f"              thickness={self._d[0]:.3f}\n"
            f"              scattering_coefficient={self._s[0]:.3f}\n"
            f"              xpd_coefficient={self._kx[0]:.3f}"
        )
        return s

    ##############################################
    # Internal methods
    ##############################################

    def _event_probabilities(
        self,
        r_te: mi.Complex2f,
        r_tm: mi.Complex2f,
        t_te: mi.Complex2f,
        t_tm: mi.Complex2f,
        enabled_interactions: mi.UInt,
    ) -> Tuple[mi.Float, mi.Float, mi.Float, mi.Bool]:

        r = dr.square(dr.abs(r_te)) + dr.square(dr.abs(r_tm))
        t = dr.square(dr.abs(t_te)) + dr.square(dr.abs(t_tm))
        s = dr.square(self._s)

        specular_reflection_enabled = enabled_interactions & InteractionType.SPECULAR > 0
        diffuse_reflection_enabled = enabled_interactions & InteractionType.DIFFUSE > 0
        refraction_enabled = enabled_interactions & InteractionType.REFRACTION > 0
        diffraction_enabled = enabled_interactions == InteractionType.DIFFRACTION

        prs = dr.select(specular_reflection_enabled, r * (1.0 - s), mi.Float(0.0))
        prd = dr.select(diffuse_reflection_enabled, r * s, mi.Float(0.0))
        pt = dr.select(refraction_enabled, t, mi.Float(0.0))
        pd = dr.select(diffraction_enabled, mi.Float(1.0), mi.Float(0.0))

        sum_probs = prs + prd + pt + pd
        none_int = sum_probs <= 0.0
        norm_factor = dr.select(none_int, 1.0, dr.rcp(sum_probs))
        prs *= norm_factor
        prd *= norm_factor
        pt *= norm_factor
        pd *= norm_factor

        return prs, prd, pt, pd

    def _sample_event_type(
        self,
        sample1: mi.Float,
        r_te: mi.Complex2f,
        r_tm: mi.Complex2f,
        t_te: mi.Complex2f,
        t_tm: mi.Complex2f,
        enabled_interactions: mi.UInt,
    ) -> Tuple[mi.UInt, mi.Float, mi.Bool, mi.Bool, mi.Bool]:

        prs, prd, pt, _ = self._event_probabilities(
            r_te, r_tm, t_te, t_tm, enabled_interactions
        )

        specular = sample1 < prs
        diffuse = (prs <= sample1) & (sample1 < prd + prs)
        transmission = sample1 >= prd + prs

        sampled_event = dr.select(specular, InteractionType.SPECULAR, InteractionType.DIFFUSE)
        sampled_event[transmission] = InteractionType.REFRACTION

        probs = dr.select(specular, prs, prd)
        probs[transmission] = pt

        return sampled_event, probs, specular, diffuse, transmission

    def _specular_reflection_transmission_direction(
        self,
        ki_local: mi.Vector3f,
        reflection: mi.Bool,
    ) -> mi.Vector3f:

        ko_local_spec_refl = mi.reflect(-ki_local)
        ko_local_trans = ki_local
        ko_local = dr.select(reflection, ko_local_spec_refl, ko_local_trans)

        return ko_local

    def _specular_reflection_transmission_matrix(
        self,
        to_world: mi.Matrix3f,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
        reflection: mi.Bool,
        r_te: mi.Complex2f,
        r_tm: mi.Complex2f,
        t_te: mi.Complex2f,
        t_tm: mi.Complex2f,
    ) -> mi.Matrix4f:

        c1 = dr.select(reflection, r_te, t_te)
        c2 = dr.select(reflection, r_tm, t_tm)

        jones_mat = jones_matrix_to_world_implicit(c1, c2, to_world, ki_local, ko_local)

        return jones_mat

    def _diffuse_reflection_direction(
        self,
        sample2: mi.Point2f,
    ) -> mi.Vector3f:

        ko_local = mi.warp.square_to_uniform_hemisphere(sample2)
        ko_local.z = dr.abs(ko_local.z)
        return ko_local

    def _diffuse_reflection_matrix(
        self,
        si: mi.SurfaceInteraction3f,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
        specular_reflection_mat: mi.Matrix4f,
    ) -> mi.Matrix4f:

        ei = mi.Vector4f(
            si.duv_dx.x,
            si.duv_dx.y,
            si.duv_dy.x,
            si.duv_dy.y,
        )
        solid_angle = si.t

        er_spec = specular_reflection_mat @ ei
        er_spec_norm = dr.norm(er_spec)
        ei_norm = dr.norm(ei)
        gamma = dr.select(ei_norm > 0.0, er_spec_norm * dr.rcp(ei_norm), mi.Float(0.0))

        fs = self._scattering_pattern(ki_local, ko_local)
        jones_mat = dr.sqrt(fs * solid_angle) * gamma * self._xpd_jones_mat

        return jones_mat

    def _diffraction_matrix(
        self,
        to_world: mi.Matrix3f,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
        si: mi.SurfaceInteraction3f,
        eta: mi.Complex2f,
        wavenumber: mi.Float,
    ) -> mi.Matrix4f:

        e_hat_local = si.dn_du
        nn_local = si.dn_dv
        n0_local = mi.Vector3f(0, 0, 1)

        wedge_angle = wedge_interior_angle(n0_local, nn_local)
        beta0 = dr.safe_acos(dr.abs(dr.dot(ki_local, e_hat_local)))
        exterior_angle = dr.two_pi - wedge_angle

        s = si.dp_du.x
        s_prime = si.dp_du.y

        to_hat_local = dr.normalize(dr.cross(n0_local, e_hat_local))

        ki_local_proj = ki_local - dr.dot(ki_local, e_hat_local) * e_hat_local
        ki_local_proj = dr.normalize(ki_local_proj)

        ko_local_proj = ko_local - dr.dot(ko_local, e_hat_local) * e_hat_local
        ko_local_proj = dr.normalize(ko_local_proj)

        phi_prime = dr.pi - dr.safe_acos(-dr.dot(ki_local_proj, to_hat_local))
        phi_prime *= -dr.sign(-dr.dot(ki_local_proj, n0_local))
        phi_prime += dr.pi

        phi = dr.pi - dr.safe_acos(dr.dot(ko_local_proj, to_hat_local))
        phi *= -dr.sign(dr.dot(ko_local_proj, n0_local))
        phi += dr.pi

        l = s * s_prime * dr.rcp(s + s_prime) * dr.sin(beta0) ** 2

        n = exterior_angle * dr.rcp(dr.pi)
        dif_phi = phi - phi_prime
        sum_phi = phi + phi_prime

        def a_p_m(beta):
            n_p = dr.round((beta + dr.pi) * dr.rcp(2 * exterior_angle))
            n_m = dr.round((beta - dr.pi) * dr.rcp(2 * exterior_angle))
            a_p = 2 * dr.cos(exterior_angle * n_p - beta / 2) ** 2
            a_m = 2 * dr.cos(exterior_angle * n_m - beta / 2) ** 2
            return a_p, a_m

        a1, a2 = a_p_m(dif_phi)
        a3, a4 = a_p_m(sum_phi)

        factor = -dr.exp(mi.Complex2f(0, -dr.pi / 4))
        factor *= dr.rcp(2 * n * dr.safe_sqrt(dr.two_pi * wavenumber) * dr.sin(beta0))

        d1 = cot((dr.pi + dif_phi) * dr.rcp(2 * n))
        d2 = cot((dr.pi - dif_phi) * dr.rcp(2 * n))
        d3 = cot((dr.pi + sum_phi) * dr.rcp(2 * n))
        d4 = cot((dr.pi - sum_phi) * dr.rcp(2 * n))

        d1 *= factor * f_utd(wavenumber * l * a1)
        d2 *= factor * f_utd(wavenumber * l * a2)
        d3 *= factor * f_utd(wavenumber * l * a3)
        d4 *= factor * f_utd(wavenumber * l * a4)

        phi_hat_prime = dr.normalize(dr.cross(ki_local, e_hat_local))
        phi_hat = -dr.normalize(dr.cross(ko_local, e_hat_local))

        e_i_s_0_hat = dr.normalize(dr.cross(ki_local, n0_local))
        e_r_s_0_hat = e_i_s_0_hat

        e_i_s_n_hat = dr.normalize(dr.cross(ki_local, nn_local))
        e_r_s_n_hat = e_i_s_n_hat

        w_0_in = jones_matrix_rotator(ki_local, phi_hat_prime, e_i_s_0_hat)
        w_0_out = jones_matrix_rotator(ko_local, e_r_s_0_hat, phi_hat)
        w_n_in = jones_matrix_rotator(ki_local, phi_hat_prime, e_i_s_n_hat)
        w_n_out = jones_matrix_rotator(ko_local, e_r_s_n_hat, phi_hat)

        r_te_0, r_tm_0, _, _ = itu_coefficients_single_layer_slab(
            dr.abs(dr.sin(phi_prime)), eta, self._d, wavenumber
        )
        r_te_n, r_tm_n, _, _ = itu_coefficients_single_layer_slab(
            dr.abs(dr.sin(exterior_angle - phi)), eta, self._d, wavenumber
        )

        d12 = -(d1 + d2)
        r_te_0 *= d4
        r_tm_0 *= d4
        r_te_n *= d3
        r_tm_n *= d3

        r_0_real = mi.Matrix2f(r_te_0.real, 0, 0, r_tm_0.real)
        r_0_imag = mi.Matrix2f(r_te_0.imag, 0, 0, r_tm_0.imag)
        r_n_real = mi.Matrix2f(r_te_n.real, 0, 0, r_tm_n.real)
        r_n_imag = mi.Matrix2f(r_te_n.imag, 0, 0, r_tm_n.imag)

        r_0_real = w_0_out @ r_0_real @ w_0_in
        r_0_imag = w_0_out @ r_0_imag @ w_0_in
        r_n_real = w_n_out @ r_n_real @ w_n_in
        r_n_imag = w_n_out @ r_n_imag @ w_n_in

        d_12_real = mi.Matrix2f(d12.real, 0, 0, d12.real)
        d_12_imag = mi.Matrix2f(d12.imag, 0, 0, d12.imag)

        real = d_12_real + r_0_real + r_n_real
        imag = d_12_imag + r_0_imag + r_n_imag

        ki_world = to_world @ ki_local
        ko_world = to_world @ ko_local
        theta_i_world = implicit_basis_vector(ki_world)
        theta_o_world = implicit_basis_vector(ko_world)
        theta_i_local = to_world.T @ theta_i_world
        theta_o_local = to_world.T @ theta_o_world
        w_in = jones_matrix_rotator(ki_local, theta_i_local, phi_hat_prime)
        w_out = jones_matrix_rotator(ko_local, phi_hat, theta_o_local)

        real = w_out @ real @ w_in
        imag = w_out @ imag @ w_in

        m4f = mi.Matrix4f(
            real[0, 0], real[0, 1], -imag[0, 0], -imag[0, 1],
            real[1, 0], real[1, 1], -imag[1, 0], -imag[1, 1],
            imag[0, 0], imag[0, 1], real[0, 0], real[0, 1],
            imag[1, 0], imag[1, 1], real[1, 0], real[1, 1],
        )

        return m4f

    def _build_xpd_jones_mat(self):
        a = dr.sqrt(1.0 - self._kx)
        b = dr.sqrt(self._kx)

        m = mi.Matrix4f(
            a, -b, 0.0, 0.0,
            b,  a, 0.0, 0.0,
            0.0, 0.0, a, -b,
            0.0, 0.0, b,  a,
        )
        self._xpd_jones_mat = m

    def _diffraction_direction(
        self,
        si: mi.SurfaceInteraction3f,
        sample: mi.Float,
        ki_local: mi.Vector3f,
    ) -> mi.Vector3f:

        e_hat_local = si.dn_du
        nn_local = si.dn_dv
        n0_local = mi.Vector3f(0.0, 0.0, 1.0)

        k_diffr = sample_keller_cone(e_hat_local, n0_local, nn_local, sample, ki_local, True)

        return k_diffr

    def _build_mi_props_from_params(
        self,
        name: str,
        thickness: float | mi.Float,
        relative_permittivity: float | mi.Float,
        conductivity: float | mi.Float,
        scattering_coefficient: float | mi.Float,
        xpd_coefficient: float | mi.Float,
        color: Tuple[float, float, float] | None,
        **kwargs,
    ) -> mi.Properties:

        props = mi.Properties("radio-material")
        props.set_id(name)

        props["relative_permittivity"] = relative_permittivity
        props["conductivity"] = conductivity
        props["scattering_coefficient"] = scattering_coefficient
        props["thickness"] = thickness
        props["xpd_coefficient"] = xpd_coefficient
        if color is not None:
            props["color"] = mi.ScalarColor3f(color)

        for k, v in kwargs.items():
            props[k] = v

        return props


mi.register_bsdf("radio-material", lambda props: RadioMaterial(props=props))