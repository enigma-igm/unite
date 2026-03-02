"""
Balmer Absorption Module

Computes Voigt absorption profiles for HI Balmer lines
using the Humlicek (1982) W4 rational approximation to the Faddeeva function.
All functions are JAX-JITable.

Includes full Balmer series (up to n=50) and bound-free continuum
for post-fit transmission modelling.
"""

from typing import Final, List, Tuple

from jax import jit, numpy as jnp

# Physical constants (CGS)
E_CGS: Final[float] = 4.8032047e-10  # electron charge (statcoulomb)
ME_CGS: Final[float] = 9.1093837e-28  # electron mass (g)
C_CGS: Final[float] = 2.99792458e10  # speed of light (cm/s)
C_KMS: Final[float] = 2.99792458e5  # speed of light (km/s)

# Rydberg constant for hydrogen (cm⁻¹)
R_H: Final[float] = 1.09677583e5

# Balmer line constants: (rest wavelength Å, oscillator strength, damping rate s⁻¹)
# γ = γ(n=2) + γ(n_upper), where γ(n) = Σ A(n→j) for all j < n
# γ(n=2) = 6.2649e8, γ(n=3) = 2.1135e8, γ(n=4) = 8.5585e7
HALPHA: Final[tuple] = (6564.61, 0.6407, 8.378e8)
HBETA: Final[tuple] = (4862.68, 0.1193, 7.121e8)

# Balmer series limit (Å) and bound-free cross-section at threshold (cm²)
BALMER_LIMIT: Final[float] = 4.0 / R_H * 1e8  # ≈ 3647 Å
SIGMA_BF_BALMER: Final[float] = 1.26e-17

# Tabulated oscillator strengths for Balmer series n=3..20 (NIST / Wiese et al.)
_BALMER_FOSC = {
    3: 0.6407, 4: 0.1193, 5: 0.04467, 6: 0.02209,
    7: 0.01270, 8: 0.008036, 9: 0.005429, 10: 0.003862,
    11: 0.002851, 12: 0.002174, 13: 0.001695, 14: 0.001346,
    15: 0.001088, 16: 0.000893, 17: 0.000742, 18: 0.000624,
    19: 0.000530, 20: 0.000454,
}


def balmer_line_data(n_max: int = 50) -> List[Tuple[float, float, float]]:
    """
    Return Balmer line data (λ_rest, f_osc, γ) for n=3 to n_max.

    Wavelengths from Rydberg formula. Oscillator strengths from NIST for
    n ≤ 20, extrapolated as f ∝ n⁻³ for n > 20.  Damping rates use known
    values for Hα/Hβ; for n ≥ 5, γ ≈ γ(n=2) since A_total(n) is small.
    """
    lines = []
    for n in range(3, n_max + 1):
        # Wavelength (Å)
        lam = 1e8 / (R_H * (0.25 - 1.0 / n**2))

        # Oscillator strength
        if n in _BALMER_FOSC:
            f = _BALMER_FOSC[n]
        else:
            f = _BALMER_FOSC[20] * (20.0 / n) ** 3

        # Damping constant
        if n == 3:
            gamma = 8.378e8
        elif n == 4:
            gamma = 7.121e8
        else:
            gamma = 6.265e8  # dominated by γ(n=2) for n ≥ 5

        lines.append((lam, f, gamma))
    return lines


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


def full_balmer_transmission(wave_obs, z_source, log_NHI, b_abs_kms, delta_v_kms,
                             n_max=50):
    """
    Full Balmer transmission: lines n=3..n_max plus bound-free continuum.

    Uses a Python loop over lines so NOT suited for JIT / MCMC.
    Intended for post-fit plotting with fixed parameters.

    Parameters
    ----------
    wave_obs : array
        Observed wavelength (Å).
    z_source, log_NHI, b_abs_kms, delta_v_kms : float
        Same as ``balmer_transmission``.
    n_max : int
        Highest Balmer line to include (default 50).

    Returns
    -------
    jnp.ndarray
        Transmission T(λ).
    """
    NHI = 10.0 ** log_NHI
    z_abs = (1 + z_source) * (1 + delta_v_kms / C_KMS) - 1

    # Bound-bound lines
    total_sigma = jnp.zeros_like(wave_obs)
    for lam_rest, f_osc, gamma in balmer_line_data(n_max):
        center_obs = lam_rest * (1 + z_abs)
        sigma = voigt_cross_section(wave_obs, center_obs, b_abs_kms, gamma, f_osc)
        total_sigma = total_sigma + sigma

    # Bound-free: photoionization from n=2
    lam_limit_obs = BALMER_LIMIT * (1 + z_abs)
    bf_sigma = SIGMA_BF_BALMER * (wave_obs / lam_limit_obs) ** 3
    bf_sigma = jnp.where(wave_obs <= lam_limit_obs, bf_sigma, 0.0)
    total_sigma = total_sigma + bf_sigma

    tau = NHI * total_sigma
    return jnp.exp(-tau)
