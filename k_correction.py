"""
k_correction.py
===============
Compute the UV k-correction and absolute UV magnitude (M_1500 or M_1700)
for i-dropout LBG candidates at z ~ 6.9.

Method follows Ono+2004 (O04) and Zewdie+2023, adapted for z ~ 6.9.

For i-dropouts at z ~ 6.9:
    - Rest-frame 1500 Å redshifts to ~13,350 Å  (J-band)
    - Rest-frame 1700 Å redshifts to ~15,230 Å  (H-band)
    - The Y-band pivot (~10,000 Å) probes rest-frame ~1266 Å
    → Use Y-band as the detection band (CTIO_DECam.Y.dat)

Dependencies
------------
    numpy, astropy, synphot
    inoue_igm_2.py  (must be in the same directory or on PYTHONPATH)

Usage
-----
    from k_correction import compute_kcorrection, compute_M_UV

    k, info = compute_kcorrection(wave_rest, flux_rest, filter_file,
                                  z=6.9, lambda_ref=1500)
    M, d_L  = compute_M_UV(m_obs=26.5, z=6.9, k=k)
"""

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from synphot import SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D

from inoue_igm_2 import inoue_igm_model

# ---------------------------------------------------------------------------
# Default cosmology  (H0=70, Om0=0.3, flat ΛCDM)
# ---------------------------------------------------------------------------
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


# ---------------------------------------------------------------------------
# Helper: load a two-column filter throughput file
# ---------------------------------------------------------------------------
def load_filter(filename):
    """
    Load a DECam (or similar) filter throughput file.

    Parameters
    ----------
    filename : str
        Path to a two-column ASCII file: wavelength [Å], throughput.

    Returns
    -------
    SpectralElement
    """
    w, t = np.loadtxt(filename, unpack=True)
    idx  = np.argsort(w)
    return SpectralElement(
        Empirical1D,
        points=w[idx] * u.AA,
        lookup_table=t[idx],
    )


# ---------------------------------------------------------------------------
# Helper: build a top-hat bandpass
# ---------------------------------------------------------------------------
def tophat_bandpass(lambda_center, half_width=5.0):
    """
    Create a top-hat SpectralElement centred on *lambda_center* [Å].

    Parameters
    ----------
    lambda_center : float
        Central rest-frame wavelength in Å (e.g. 1500 or 1700).
    half_width : float
        Half-width of the top-hat in Å (default 5 Å, matching O04).

    Returns
    -------
    SpectralElement
    """
    w = np.array([
        lambda_center - half_width - 1.0,
        lambda_center - half_width,
        lambda_center + half_width,
        lambda_center + half_width + 1.0,
    ])
    t = np.array([0.0, 1.0, 1.0, 0.0])
    return SpectralElement(Empirical1D, points=w * u.AA, lookup_table=t)


# ---------------------------------------------------------------------------
# Helper: synthetic AB magnitude through a filter
# ---------------------------------------------------------------------------
def synthetic_abmag(source, filt):
    """Return synthetic AB magnitude of *source* through *filt*."""
    obs = Observation(source, filt, force="taper")
    return obs.effstim("abmag").value


# ---------------------------------------------------------------------------
# Core function 1: k-correction
# ---------------------------------------------------------------------------
def compute_kcorrection(
    wave_rest,
    flux_rest,
    obs_filter,
    z,
    lambda_ref=1500,
    half_width=5.0,
    apply_igm=True,
):
    """
    Compute the k-correction between the observed-frame detection band
    and a top-hat at *lambda_ref* Å in the rest frame.

        k  =  m_obs_band  −  m_tophat_restframe

    so that:
        M_UV  =  m_obs  +  2.5·log10(1+z)  −  5·log10(d_L [Mpc])  −  25  −  k

    Parameters
    ----------
    wave_rest : array_like
        Rest-frame wavelengths of the LBG template [Å].
    flux_rest : array_like
        Corresponding flux densities (arbitrary units; only ratios matter).
    obs_filter : SpectralElement or str
        The observed-frame detection filter.  Pass either a pre-loaded
        SpectralElement or a path string to a throughput file.
    z : float
        Source redshift.
    lambda_ref : float
        Rest-frame reference wavelength in Å.  Use 1500 or 1700.
    half_width : float
        Half-width of the top-hat bandpass in Å (default 5 Å).
    apply_igm : bool
        Apply Inoue+2014 IGM attenuation to the redshifted template
        before computing the observed-frame magnitude.

    Returns
    -------
    k : float
        k-correction in magnitudes.
    info : dict
        Dictionary with intermediate results:
        {
          "m_obs_band"      : synthetic AB mag in the detection band,
          "m_tophat"        : synthetic AB mag in the rest-frame top-hat,
          "lambda_eff_rest" : effective rest-frame wavelength of the
                              detection band at this redshift [Å],
          "lambda_ref"      : reference wavelength used [Å],
          "z"               : redshift,
        }
    """
    wave_rest = np.asarray(wave_rest, dtype=float)
    flux_rest = np.asarray(flux_rest, dtype=float)

    # ------------------------------------------------------------------
    # 1. Normalize template at lambda_ref, then scale to AB=0 at lambda_ref
    # ------------------------------------------------------------------
    f_norm = np.interp(lambda_ref, wave_rest, flux_rest)
    if f_norm <= 0 or not np.isfinite(f_norm):
        raise ValueError(
            f"Template flux at lambda_ref={lambda_ref} Å is non-positive "
            "or non-finite. Check your template wavelength range."
        )

    flux_rest = flux_rest / f_norm          # unit flux at lambda_ref

    # f_lambda [erg/s/cm²/Å] corresponding to AB mag = 0 at lambda_ref
    # f_nu(AB=0) = 3.631e-20 erg/s/cm²/Hz  →  f_lambda = f_nu * c / lambda²
    c_AAs = 2.99792458e18                  # speed of light in Å/s
    f_lambda_AB0 = (3.631e-20 * c_AAs) / (lambda_ref ** 2)

    flux_rest = flux_rest * f_lambda_AB0   # now in erg/s/cm²/Å  (AB=0 at lambda_ref)

    # ------------------------------------------------------------------
    # 2. Load filter if a filename string was given
    # ------------------------------------------------------------------
    if isinstance(obs_filter, str):
        obs_filter = load_filter(obs_filter)

    # ------------------------------------------------------------------
    # 3. Build the observed-frame wavelength grid (covers the filter)
    # ------------------------------------------------------------------
    w_obs = np.linspace(
        obs_filter.waveset.min().value - 500,
        obs_filter.waveset.max().value + 500,
        12000,
    )

    # ------------------------------------------------------------------
    # 4. Redshift the rest-frame template onto the observed-frame grid
    # ------------------------------------------------------------------
    w_rest_z  = w_obs / (1.0 + z)
    flux_obs  = np.interp(w_rest_z, wave_rest, flux_rest, left=0.0, right=0.0)
    flux_obs /= (1.0 + z)           # (1+z) bandwidth compression / flux dilution

    # ------------------------------------------------------------------
    # 5. Apply IGM attenuation (Inoue+2014)
    # ------------------------------------------------------------------
    if apply_igm:
        T_igm    = inoue_igm_model(w_obs, z_s=z, output="transmission")
        flux_obs = flux_obs * T_igm

    flux_obs = np.clip(flux_obs, 0.0, None)

    if flux_obs.max() == 0:
        raise ValueError(
            "Template has zero flux after IGM attenuation inside the "
            "detection filter.  Check the wavelength coverage of your "
            "template and that it extends to the observed-frame band."
        )

    # ------------------------------------------------------------------
    # 6. Build SourceSpectrum in the observed frame
    # ------------------------------------------------------------------
    src_obs = SourceSpectrum(
        Empirical1D,
        points=w_obs * u.AA,
        lookup_table=flux_obs * u.Unit("erg s-1 cm-2 AA-1"),
    )

    # ------------------------------------------------------------------
    # 7. Synthetic AB magnitude through the detection filter
    # ------------------------------------------------------------------
    m_obs_band = synthetic_abmag(src_obs, obs_filter)

    # ------------------------------------------------------------------
    # 8. Rest-frame magnitude through the top-hat at lambda_ref
    #    (no redshift, no IGM — pure rest-frame)
    # ------------------------------------------------------------------
    good = (flux_rest > 0) & np.isfinite(flux_rest)
    src_rest = SourceSpectrum(
        Empirical1D,
        points=wave_rest[good] * u.AA,
        lookup_table=flux_rest[good] * u.Unit("erg s-1 cm-2 AA-1"),
    )
    tophat   = tophat_bandpass(lambda_ref, half_width=half_width)
    m_tophat = synthetic_abmag(src_rest, tophat)

    # ------------------------------------------------------------------
    # 9. k-correction  =  m_obs_band − m_tophat
    # ------------------------------------------------------------------
    k = m_obs_band - m_tophat

    # Effective rest-frame wavelength probed by the detection filter
    lambda_eff_obs  = obs_filter.pivot().value      # pivot λ [Å] in observed frame
    lambda_eff_rest = lambda_eff_obs / (1.0 + z)   # corresponding rest-frame λ [Å]

    info = {
        "m_obs_band"      : m_obs_band,
        "m_tophat"        : m_tophat,
        "lambda_eff_rest" : lambda_eff_rest,
        "lambda_ref"      : lambda_ref,
        "z"               : z,
    }

    return k, info


# ---------------------------------------------------------------------------
# Core function 2: absolute UV magnitude
# ---------------------------------------------------------------------------
def compute_M_UV(m_obs, z, k, cosmo=None):
    """
    Convert apparent magnitude to absolute UV magnitude.

    Uses Equation (5) of Zewdie+2023 / O04, with d_L in Mpc:

        M_UV = m_obs + 2.5·log10(1+z) − 5·log10(d_L [Mpc]) − 25 − k

    Derivation of the −25 constant
    --------------------------------
    The standard distance modulus is  μ = 5·log10(d [pc] / 10).
    Converting d_L from Mpc to pc:
        d [pc] = d_L [Mpc] × 10^6
        μ = 5·log10(d_L [Mpc] × 10^6 / 10)
          = 5·log10(d_L [Mpc]) + 5·log10(10^5)
          = 5·log10(d_L [Mpc]) + 25
    Therefore:
        M = m − μ = m − 5·log10(d_L [Mpc]) − 25

    Parameters
    ----------
    m_obs : float or array_like
        Apparent magnitude(s) in the detection band.
    z : float
        Source redshift.
    k : float
        k-correction (from compute_kcorrection).
    cosmo : astropy.cosmology instance, optional
        Defaults to FlatLambdaCDM(H0=70, Om0=0.3).

    Returns
    -------
    M_UV : float or ndarray
        Absolute UV magnitude(s).
    d_L_Mpc : float
        Luminosity distance in Mpc (informational).
    """
    if cosmo is None:
        cosmo = COSMO

    d_L_Mpc = cosmo.luminosity_distance(z).to(u.Mpc).value

    # KEY: constant is −25 when d_L is in Mpc (not +5)
    # +5 is only correct when d_L is already in pc and divided by 10.
    M_UV = (
        np.asarray(m_obs)
        + 2.5 * np.log10(1.0 + z)
        - 5.0 * np.log10(d_L_Mpc)
        - 25.0                          # ← correct constant for d_L in Mpc
        - k
    )

    return M_UV, d_L_Mpc


# ---------------------------------------------------------------------------
# Convenience wrapper: do everything in one call
# ---------------------------------------------------------------------------
def kcorrect_and_M_UV(
    wave_rest,
    flux_rest,
    obs_filter,
    m_obs,
    z=6.9,
    lambda_ref=1500,
    half_width=5.0,
    apply_igm=True,
    cosmo=None,
    verbose=True,
):
    """
    One-stop function: compute the k-correction and absolute UV magnitude.

    Parameters
    ----------
    wave_rest, flux_rest : array_like
        Shapley+03 (or other) LBG template in rest-frame wavelength [Å]
        and flux density.
    obs_filter : SpectralElement or str
        Detection-band filter (SpectralElement or path to file).
        For i-dropouts at z~6.9 use the Y-band: 'CTIO_DECam.Y.dat'
    m_obs : float or array_like
        Apparent magnitude(s) of your LBG candidate(s) in the detection band.
    z : float
        Source / selection redshift (default 6.9).
    lambda_ref : {1500, 1700}
        Rest-frame UV reference wavelength in Å.
    half_width : float
        Half-width of the top-hat bandpass in Å (default 5 Å).
    apply_igm : bool
        Apply IGM attenuation (default True).
    cosmo : astropy cosmology, optional
    verbose : bool
        Print a summary table (default True).

    Returns
    -------
    k : float
        k-correction [mag].
    M_UV : float or ndarray
        Absolute UV magnitude(s).
    info : dict
        Diagnostic quantities (m_obs_band, m_tophat, lambda_eff_rest,
        lambda_ref, z, d_L_Mpc).
    """
    k, info = compute_kcorrection(
        wave_rest, flux_rest, obs_filter,
        z=z, lambda_ref=lambda_ref,
        half_width=half_width, apply_igm=apply_igm,
    )

    M_UV, d_L_Mpc = compute_M_UV(m_obs, z, k, cosmo=cosmo)
    info["d_L_Mpc"] = d_L_Mpc

    if verbose:
        print("=" * 60)
        print(f"  k-correction summary  (λ_ref = {lambda_ref} Å,  z = {z})")
        print("=" * 60)
        print(f"  Detection band rest-frame pivot λ : {info['lambda_eff_rest']:.1f} Å")
        print(f"  m_obs_band (template in filter)   : {info['m_obs_band']:.4f}")
        print(f"  m_tophat   (top-hat at λ_ref)     : {info['m_tophat']:.4f}")
        print(f"  k-correction                      : {k:+.4f} mag")
        print(f"  Luminosity distance d_L           : {d_L_Mpc:,.1f} Mpc")
        print("-" * 60)
        for mo, mu in zip(
            np.atleast_1d(m_obs), np.atleast_1d(M_UV)
        ):
            print(f"  m_obs = {mo:.2f}  →  M_{lambda_ref} = {mu:.3f}")
        print("=" * 60)

    return k, M_UV, info


# ---------------------------------------------------------------------------
# Quick demo / sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    print("\n--- k_correction.py  demo ---\n")

    # ── Paths: edit these to match your local files ───────────────────────
    SHAPLEY_FILE = "shapely_spectrum.txt"   # Shapley+03 LBG composite
    FILTER_FILE  = "CTIO_DECam.Y.dat"      # Y-band for i-dropouts at z~6.9
                                            # (z-band pivot is ~1160 Å rest-frame
                                            #  at z=6.9, too blue; use Y-band)
    Z_SOURCE     = 6.9                      # i-dropout selection redshift
    LAMBDA_REF   = 1500                     # 1500 or 1700 Å
    # ─────────────────────────────────────────────────────────────────────

    # ── Load or generate template ─────────────────────────────────────────
    if not os.path.exists(SHAPLEY_FILE):
        print(f"Template file not found: {SHAPLEY_FILE}")
        print("Generating a β = −2 power-law template for demonstration.\n")
        w_rest = np.linspace(900, 3000, 2000)
        f_rest = (w_rest / 1500.0) ** (-2.0)
    else:
        data   = np.genfromtxt(SHAPLEY_FILE, comments="#")
        w_rest = data[:, 0]
        f_rest = data[:, 1]
        good   = (f_rest > 0) & np.isfinite(f_rest)
        w_rest = w_rest[good]
        f_rest = f_rest[good]
        idx    = np.argsort(w_rest)
        w_rest, f_rest = w_rest[idx], f_rest[idx]

    # ── Example apparent magnitudes ───────────────────────────────────────
    m_candidates = np.array([24.5, 25.0, 25.5, 26.0, 26.5])

    # ── Sanity check on M_UV formula (no filter needed) ──────────────────
    # At z=6.9 with d_L~67883 Mpc, m=26.5, k=0:
    #   M = 26.5 + 2.5*log10(7.9) - 5*log10(67883) - 25
    #     ≈ 26.5 + 2.248 - 24.159 - 25  ≈  -20.41   ✓ (physically reasonable)
    d_L_check = COSMO.luminosity_distance(Z_SOURCE).to(u.Mpc).value
    M_check   = 26.5 + 2.5*np.log10(1+Z_SOURCE) - 5*np.log10(d_L_check) - 25.0
    print(f"Sanity check (k=0, m=26.5, z={Z_SOURCE}): M_UV = {M_check:.3f}  "
          f"(expect ~ −20.4)\n")

    # ── Full k-correction (requires filter file) ──────────────────────────
    if not os.path.exists(FILTER_FILE):
        print(f"Filter file not found: {FILTER_FILE}")
        print("Skipping full k-correction. Set FILTER_FILE to a valid path.\n")
    else:
        k, M_UV, info = kcorrect_and_M_UV(
            w_rest, f_rest,
            obs_filter = FILTER_FILE,
            m_obs      = m_candidates,
            z          = Z_SOURCE,
            lambda_ref = LAMBDA_REF,
            verbose    = True,
        )
