"""
inoue_igm.py

Inoue+2014 IGM absorption model + utilities for synthetic photometry
and colour track generation.

Author: You :)
"""

import numpy as np
from synphot import SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D
from astropy import units as u

# ============================================================
# --- IGM MODEL WRAPPER
# ============================================================
def inoue_igm_model(wave, z_s, output="transmission"):
    """
    Wrapper for Inoue+2014 IGM absorption model.

    Parameters
    ----------
    wave : array_like
        Observed wavelength(s) in Angstrom.
    z_s : float
        Source redshift.
    output : {"transmission", "tau", "tau_ls", "tau_lc"}

    Returns
    -------
    array_like
    """
    wave = np.asarray(wave)

    if output == "transmission":
        return transmission_inoue(wave, z_s)
    elif output == "tau":
        return total_tau_inoue(wave, z_s)
    elif output == "tau_ls":
        return tau_LS_inoue(wave, z_s)
    elif output == "tau_lc":
        return tau_LC_inoue(wave, z_s)
    else:
        raise ValueError("Invalid output type")


# ============================================================
# --- CONSTANTS
# ============================================================
lambda_L = 911.75  # Angstrom

lambda_j = np.array([
    1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226,
    923.150, 920.963, 919.352, 918.129, 917.181, 916.429, 915.824,
    915.329, 914.919, 914.576, 914.286, 914.039, 913.826, 913.641,
    913.480, 913.339, 913.215, 913.104, 913.006, 912.918, 912.839,
    912.768, 912.703, 912.645, 912.592, 912.543, 912.499, 912.458,
    912.420, 912.385, 912.353, 912.324
])


ALAF1 = np.array([
    1.690e-02, 4.692e-03, 2.239e-03, 1.319e-03, 8.707e-04, 6.178e-04,
    4.609e-04, 3.569e-04, 2.843e-04, 2.318e-04, 1.923e-04, 1.622e-04,
    1.385e-04, 1.196e-04, 1.043e-04, 9.174e-05, 8.128e-05, 7.251e-05,
    6.505e-05, 5.868e-05, 5.319e-05, 4.843e-05, 4.427e-05, 4.063e-05,
    3.738e-05, 3.454e-05, 3.199e-05, 2.971e-05, 2.766e-05, 2.582e-05,
    2.415e-05, 2.263e-05, 2.126e-05, 2.000e-05, 1.885e-05, 1.779e-05,
    1.682e-05, 1.593e-05, 1.510e-05
])

ALAF2 = np.array([
    2.354e-03, 6.536e-04, 3.119e-04, 1.837e-04, 1.213e-04, 8.606e-05,
    6.421e-05, 4.971e-05, 3.960e-05, 3.229e-05, 2.679e-05, 2.259e-05,
    1.929e-05, 1.666e-05, 1.453e-05, 1.278e-05, 1.132e-05, 1.010e-05,
    9.062e-06, 8.174e-06, 7.409e-06, 6.746e-06, 6.167e-06, 5.660e-06,
    5.207e-06, 4.811e-06, 4.456e-06, 4.139e-06, 3.853e-06, 3.596e-06,
    3.364e-06, 3.153e-06, 2.961e-06, 2.785e-06, 2.625e-06, 2.479e-06,
    2.343e-06, 2.219e-06, 2.103e-06
])

ALAF3 = np.array([
    1.026e-04, 2.849e-05, 1.360e-05, 8.010e-06, 5.287e-06, 3.752e-06,
    2.799e-06, 2.167e-06, 1.726e-06, 1.407e-06, 1.168e-06, 9.847e-07,
    8.410e-07, 7.263e-07, 6.334e-07, 5.571e-07, 4.936e-07, 4.403e-07,
    3.950e-07, 3.563e-07, 3.230e-07, 2.941e-07, 2.689e-07, 2.467e-07,
    2.270e-07, 2.097e-07, 1.943e-07, 1.804e-07, 1.680e-07, 1.568e-07,
    1.466e-07, 1.375e-07, 1.291e-07, 1.214e-07, 1.145e-07, 1.080e-07,
    1.022e-07, 9.673e-08, 9.169e-08
])

ADLA1 = np.array([
    1.617e-04, 1.545e-04, 1.498e-04, 1.460e-04, 1.429e-04, 1.402e-04,
    1.377e-04, 1.355e-04, 1.335e-04, 1.316e-04, 1.298e-04, 1.281e-04,
    1.265e-04, 1.250e-04, 1.236e-04, 1.222e-04, 1.209e-04, 1.197e-04,
    1.185e-04, 1.173e-04, 1.162e-04, 1.151e-04, 1.140e-04, 1.130e-04,
    1.120e-04, 1.110e-04, 1.101e-04, 1.091e-04, 1.082e-04, 1.073e-04,
    1.065e-04, 1.056e-04, 1.048e-04, 1.040e-04, 1.032e-04, 1.024e-04,
    1.017e-04, 1.009e-04, 1.002e-04
])

ADLA2 = np.array([
    5.390e-05, 5.151e-05, 4.992e-05, 4.868e-05, 4.763e-05, 4.672e-05,
    4.590e-05, 4.516e-05, 4.448e-05, 4.385e-05, 4.326e-05, 4.271e-05,
    4.218e-05, 4.168e-05, 4.120e-05, 4.075e-05, 4.031e-05, 3.989e-05,
    3.949e-05, 3.910e-05, 3.872e-05, 3.836e-05, 3.800e-05, 3.766e-05,
    3.732e-05, 3.700e-05, 3.668e-05, 3.637e-05, 3.607e-05, 3.578e-05,
    3.549e-05, 3.521e-05, 3.493e-05, 3.466e-05, 3.440e-05, 3.414e-05,
    3.389e-05, 3.364e-05, 3.339e-05
])


# ============================================================
# --- TAU FUNCTIONS
# ============================================================
def tau_LS_inoue(wave, z_s):
    wave = np.asarray(wave)
    tau = np.zeros_like(wave)

    for j in range(len(lambda_j)):
        z_j = wave / lambda_j[j] - 1.0
        mask = (z_j > 0) & (z_j < z_s)

        zj = z_j[mask]
        if len(zj) == 0:
            continue

        laf = np.zeros_like(zj)
        laf[zj < 1.2] = ALAF1[j] * (1 + zj[zj < 1.2])**1.2
        mask2 = (zj >= 1.2) & (zj < 4.7)
        laf[mask2] = ALAF2[j] * (1 + zj[mask2])**3.7
        laf[zj >= 4.7] = ALAF3[j] * (1 + zj[zj >= 4.7])**5.5

        dla = np.zeros_like(zj)
        dla[zj < 2.0] = ADLA1[j] * (1 + zj[zj < 2.0])**2.0
        dla[zj >= 2.0] = ADLA2[j] * (1 + zj[zj >= 2.0])**3.0

        tau[mask] += laf + dla

    return tau


def tau_LC_inoue(wave, z_s):
    wave = np.asarray(wave)
    tau = np.zeros_like(wave)

    mask = (wave < lambda_L * (1 + z_s)) & (wave > lambda_L)
    if not np.any(mask):
        return tau

    x = wave[mask] / lambda_L

    tau_LAF = np.zeros_like(x)
    tau_DLA = np.zeros_like(x)

    # LAF
    mask1 = x < 2.2
    mask2 = (x >= 2.2) & (x < 5.7)
    mask3 = (x >= 5.7) & (x < (1 + z_s))

    if np.any(mask1):
        x1 = x[mask1]
        tau_LAF[mask1] = (5.22e-4 * (1 + z_s)**3.4 * x1**2.1 +
                         0.325 * x1**1.2 -
                         3.14e-2 * x1**2.1)

    if np.any(mask2):
        x2 = x[mask2]
        tau_LAF[mask2] = (5.22e-4 * (1 + z_s)**3.4 * x2**2.1 +
                         0.218 * x2**2.1 -
                         2.55e-2 * x2**3.7)

    if np.any(mask3):
        x3 = x[mask3]
        tau_LAF[mask3] = 5.22e-4 * ((1 + z_s)**3.4 * x3**2.1 - x3**5.5)

    # DLA
    mask1 = x < 3.0
    mask2 = (x >= 3.0) & (x < (1 + z_s))

    if np.any(mask1):
        x1 = x[mask1]
        tau_DLA[mask1] = (0.634 +
                          4.70e-2 * (1 + z_s)**3.0 -
                          1.78e-2 * (1 + z_s)**3.3 * x1**(-0.3) -
                          0.135 * x1**2.0 -
                          0.291 * x1**(-0.3))

    if np.any(mask2):
        x2 = x[mask2]
        tau_DLA[mask2] = (4.70e-2 * (1 + z_s)**3.0 -
                          1.78e-2 * (1 + z_s)**3.3 * x2**(-0.3) -
                          2.92e-2 * x2**3.0)

    tau[mask] = tau_LAF + tau_DLA
    return tau


def total_tau_inoue(wave, z_s):
    tau_ls = tau_LS_inoue(wave, z_s)

    tau_lc = np.zeros_like(wave)
    mask = wave < lambda_L * (1 + z_s)
    if np.any(mask):
        tau_lc[mask] = tau_LC_inoue(wave[mask], z_s)

    return tau_ls + tau_lc


def transmission_inoue(wave, z_s):
    return np.exp(-total_tau_inoue(wave, z_s))


# ============================================================
# --- SYNPHOT UTILITIES
# ============================================================
def synthetic_mag(source, filt):
    obs = Observation(source, filt, force="taper")
    return obs.effstim("abmag").value


def synphot_filter(filename):
    w, t = np.loadtxt(filename, unpack=True)
    idx = np.argsort(w)

    return SpectralElement(
        Empirical1D,
        points=w[idx] * u.AA,
        lookup_table=t[idx]
    )


# ============================================================
# --- APPLY IGM
# ============================================================
def apply_inoue_igm(w_obs, flux_obs, z):
    T_igm = inoue_igm_model(w_obs, z, output="transmission")
    return flux_obs * T_igm


# ============================================================
# --- COLOUR TRACK FUNCTION
# ============================================================
def colour_track(z_vals, wave_rest, flux_rest, f_i, f_z, f_y, maglim_i=27.5):

    i_mags, z_mags, y_mags = [], [], []

    wmin = min(f_i.waveset.min(),
               f_z.waveset.min(),
               f_y.waveset.min()).value

    wmax = max(f_i.waveset.max(),
               f_z.waveset.max(),
               f_y.waveset.max()).value

    w_obs = np.linspace(wmin - 500, wmax + 500, 8000)

    for z in z_vals:

        w_rest_z = w_obs / (1 + z)
        flux_rest_z = np.interp(w_rest_z, wave_rest, flux_rest, left=0, right=0)

        flux_obs = flux_rest_z / (1 + z)
        flux_igm = apply_inoue_igm(w_obs, flux_obs, z)

        src = SourceSpectrum(
            Empirical1D,
            points=w_obs * u.AA,
            lookup_table=flux_igm * u.Unit("erg s-1 cm-2 AA-1")
        )

        mi = synthetic_mag(src, f_i)
        mz = synthetic_mag(src, f_z)
        my = synthetic_mag(src, f_y)

        mi = max(mi, maglim_i)

        i_mags.append(mi)
        z_mags.append(mz)
        y_mags.append(my)

    return np.array(i_mags), np.array(z_mags), np.array(y_mags)