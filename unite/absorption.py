"""
Balmer Absorption Module

Computes Voigt absorption profiles for HI Balmer lines (Hα, Hβ)
using the Humlicek (1982) W4 rational approximation to the Faddeeva function.
All functions are JAX-JITable.
"""

from typing import Final

from jax import jit, numpy as jnp

# Physical constants (CGS)
E_CGS: Final[float] = 4.8032047e-10  # electron charge (statcoulomb)
ME_CGS: Final[float] = 9.1093837e-28  # electron mass (g)
C_CGS: Final[float] = 2.99792458e10  # speed of light (cm/s)
C_KMS: Final[float] = 2.99792458e5  # speed of light (km/s)

# Balmer line constants: (rest wavelength Å, oscillator strength, damping rate s⁻¹)
# γ = γ(n=2) + γ(n_upper), where γ(n) = Σ A(n→j) for all j < n
# γ(n=2) = 6.2649e8, γ(n=3) = 2.1135e8, γ(n=4) = 8.5585e7
HALPHA: Final[tuple] = (6564.61, 0.6407, 8.378e8)
HBETA: Final[tuple] = (4862.68, 0.1193, 7.121e8)


@jit
def faddeeva_humlicek(z):
    """
    Humlicek (1982) W4 rational approximation to the Faddeeva function w(z).

    w(z) = exp(-z²) erfc(-iz) for complex z with Im(z) >= 0.
    Accurate to ~1e-4 relative error across the upper half-plane.

    Uses Humlicek's T = Y - iX variable convention internally.

    Parameters
    ----------
    z : jnp.ndarray (complex)
        Complex argument(s) with Im(z) >= 0.

    Returns
    -------
    jnp.ndarray (complex)
        w(z)
    """
    X = jnp.real(z)
    Y = jnp.imag(z)
    S = jnp.abs(X) + Y

    # Humlicek's internal variable: T = Y - iX  (equivalently, T = -iz)
    T = Y - 1j * X
    T2 = T * T  # = -z²

    # Region 1: S >= 15 — asymptotic continued fraction
    w1 = T * 0.5641896 / (T2 + 0.5)

    # Region 2: S >= 5.5
    w2 = T * (1.410474 + T2 * 0.5641896) / (0.75 + T2 * (3.0 + T2))

    # Region 3: Y >= 0.195|X| - 0.176 (and S < 5.5)
    w3 = (
        (16.4955 + T * (20.20933 + T * (11.96482 + T * (3.778987 + T * 0.5642236))))
        / (16.4955 + T * (38.82363 + T * (39.27121 + T * (21.69274 + T * (6.699398 + T)))))
    )

    # Region 4: small S, near real axis
    w4_rational = (
        T
        * (36183.31 - T2 * (3321.99 - T2 * (1540.787 - T2 * (219.031 - T2 * (35.7668 - T2 * (1.320522 - T2 * 0.56419))))))
        / (32066.6 - T2 * (24322.84 - T2 * (9022.228 - T2 * (2186.181 - T2 * (364.2191 - T2 * (61.57037 - T2 * (1.841439 - T2)))))))
    )
    w4 = jnp.exp(T2) - w4_rational  # exp(T²) = exp(-z²)

    # Region selection
    reg3_cond = Y >= (0.195 * jnp.abs(X) - 0.176)

    w = jnp.where(
        S >= 15, w1,
        jnp.where(S >= 5.5, w2,
            jnp.where(reg3_cond, w3, w4)),
    )

    return w


@jit
def voigt_cross_section(wave_obs, center_obs, b_kms, gamma, f_osc):
    """
    Compute the Voigt absorption cross-section σ(λ) for a single line.

    Parameters
    ----------
    wave_obs : jnp.ndarray
        Observed wavelength array (Å).
    center_obs : float
        Observed line center (Å).
    b_kms : float
        Doppler b parameter (km/s).
    gamma : float
        Damping constant (s⁻¹).
    f_osc : float
        Oscillator strength.

    Returns
    -------
    jnp.ndarray
        Cross-section σ in cm² (evaluated at each wavelength).
    """
    # Convert to frequency space
    nu0 = C_CGS / (center_obs * 1e-8)  # line center frequency (Hz)
    nu = C_CGS / (wave_obs * 1e-8)  # frequency array (Hz)

    # Doppler width in frequency
    b_cms = b_kms * 1e5  # km/s -> cm/s
    delta_nu_D = nu0 * b_cms / C_CGS

    # Voigt parameters
    a = gamma / (4 * jnp.pi * delta_nu_D)  # damping parameter
    u = (nu - nu0) / delta_nu_D  # normalized detuning

    # Faddeeva function: H(a, u) = Re[w(u + ia)]
    z = u + 1j * a
    H = jnp.real(faddeeva_humlicek(z))

    # Voigt profile normalized in frequency: φ(ν) = H(a,u) / (√π Δν_D)
    phi_nu = H / (jnp.sqrt(jnp.pi) * delta_nu_D)

    # Cross-section: σ(ν) = (π e² / m_e c) × f × φ(ν)
    # Units: [cm² Hz] × [dimensionless] × [Hz⁻¹] = [cm²]
    sigma = (jnp.pi * E_CGS**2 / (ME_CGS * C_CGS)) * f_osc * phi_nu

    return sigma


@jit
def balmer_transmission(wave_obs, z_source, log_NHI, b_abs_kms, delta_v_kms):
    """
    Compute the Balmer absorption transmission T(λ) for Hα and Hβ.

    T(λ) = exp(-N_HI × Σᵢ σᵢ(λ))

    Parameters
    ----------
    wave_obs : jnp.ndarray
        Observed wavelength array (Å).
    z_source : float
        Source redshift.
    log_NHI : float
        log₁₀(N_HI / cm⁻²).
    b_abs_kms : float
        Doppler b parameter (km/s).
    delta_v_kms : float
        Velocity offset of absorber relative to source (km/s).

    Returns
    -------
    jnp.ndarray
        Transmission T(λ), values between 0 and 1.
    """
    # Column density
    NHI = 10.0**log_NHI

    # Absorber redshift: z_abs ≈ (1 + z_source)(1 + Δv/c) - 1
    z_abs = (1 + z_source) * (1 + delta_v_kms / C_KMS) - 1

    # Compute cross-sections for each Balmer line
    total_sigma = jnp.zeros_like(wave_obs)
    for lam_rest, f_osc, gamma in [HALPHA, HBETA]:
        center_obs = lam_rest * (1 + z_abs)
        sigma = voigt_cross_section(wave_obs, center_obs, b_abs_kms, gamma, f_osc)
        total_sigma = total_sigma + sigma

    # Optical depth and transmission
    tau = NHI * total_sigma
    transmission = jnp.exp(-tau)

    return transmission
