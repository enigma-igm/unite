#!/usr/bin/env python
"""
Plot bluejay-south-v4-12020 spectrum with Paschen and HeI absorption features.

This source has dWAIC = -48.7 for Balmer absorption (strongly preferred),
and also shows possible Paschen and HeI absorption that is NOT modeled
by the current UNITE absorption model (Balmer-only).

Panels:
  - Top: Full multi-grating spectrum with all line positions marked
  - Bottom row: Zoomed panels around Pa-delta, Pa-gamma+HeI, Pa-beta
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = '/Users/jiamuh/python/unite_fitting_data'
SPEC_DIR = os.path.join(DATA_DIR, 'dja_all_spectra')
OUTPUT_DIR = os.path.join(DATA_DIR, 'out_dja_absorption', 'Plots')

SPEC_FILES = {
    'G140M': 'bluejay-south-v4_g140m-f100lp_1810_12020.spec.fits',
    'G235M': 'bluejay-south-v4_g235m-f170lp_1810_12020.spec.fits',
    'G395M': 'bluejay-south-v4_g395m-f290lp_1810_12020.spec.fits',
}

Z = 2.093  # redshift from fit

# Absorption fit results (from broad_abs_summary.csv)
LOG_NHI = 12.13   # N(n=2) column density
B_ABS = 189.1     # km/s
DV_ABS = 741.4    # km/s
DWAIC = -48.7     # absorption model preferred

# Line wavelengths in vacuum Angstroms (rest frame)
BALMER_LINES = {
    r'H$\alpha$': 6564.61,
    r'H$\beta$': 4862.68,
    r'H$\gamma$': 4341.68,
    r'H$\delta$': 4102.89,
}

PASCHEN_LINES = {
    r'Pa$\alpha$': 18756.1,
    r'Pa$\beta$': 12821.6,
    r'Pa$\gamma$': 10941.1,
    r'Pa$\delta$': 10052.1,
    r'Pa$\epsilon$': 9546.0,
    r'Pa 9': 9229.0,
    r'Pa 10': 9015.0,
}

HELIUM_LINES = {
    r'HeI 10833': 10833.0,
    r'HeI 5877': 5877.0,
}

FORBIDDEN_LINES = {
    r'[OIII]': 5008.24,
    r'[NII]': 6585.27,
}

# Colors per grating
GRATING_COLORS = {
    'G140M': 'C0',
    'G235M': 'C1',
    'G395M': 'C2',
}


# =============================================================================
# Helpers
# =============================================================================

def load_spectrum(grating):
    """Load SPEC1D from a grating file. Returns (wave_um, flux_ujy, err_ujy)."""
    path = os.path.join(SPEC_DIR, SPEC_FILES[grating])
    spec = Table.read(path, 'SPEC1D')
    wave = np.array(spec['wave'])   # microns
    flux = np.array(spec['flux'])   # microJy
    err = np.array(spec['err'])     # microJy
    # Mask bad pixels
    good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    return wave[good], flux[good], err[good]


def mark_lines(ax, lines, z, color, ls='--', alpha=0.6, ytext=None,
               fontsize=8, offset_idx=0):
    """Mark absorption/emission lines on an axis."""
    ylim = ax.get_ylim()
    if ytext is None:
        ytext = ylim[1] * 0.92
    for i, (name, lam_rest) in enumerate(lines.items()):
        lam_obs = lam_rest * (1 + z) / 1e4  # microns
        xlim = ax.get_xlim()
        if lam_obs < xlim[0] or lam_obs > xlim[1]:
            continue
        ax.axvline(lam_obs, ls=ls, color=color, alpha=alpha, lw=1)
        # Alternate vertical positions to avoid overlap
        y_pos = ylim[1] - (0.05 + 0.06 * ((i + offset_idx) % 3)) * (ylim[1] - ylim[0])
        ax.text(lam_obs, y_pos, name, ha='center', va='top',
                fontsize=fontsize, color=color, rotation=90,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))


# =============================================================================
# Main
# =============================================================================

def main():
    print(f'Plotting bluejay-south-v4-12020 (z={Z})')
    print(f'  Balmer absorption: log_NHI={LOG_NHI}, b={B_ABS} km/s, '
          f'dv={DV_ABS} km/s, dWAIC={DWAIC}')

    # Load all gratings
    spectra = {}
    for g in SPEC_FILES:
        spectra[g] = load_spectrum(g)
        print(f'  {g}: {len(spectra[g][0])} pixels, '
              f'{spectra[g][0].min():.3f}-{spectra[g][0].max():.3f} um')

    # ---- Figure layout ----
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
    })

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

    # Top panel: full spectrum
    ax_full = fig.add_subplot(gs[0, :])

    # Bottom panels: zoomed regions
    ax_pad = fig.add_subplot(gs[1, 0])   # Pa-delta + Pa-epsilon
    ax_pag = fig.add_subplot(gs[1, 1])   # Pa-gamma + HeI
    ax_pab = fig.add_subplot(gs[1, 2])   # Pa-beta

    # ---- Full spectrum ----
    for g, color in GRATING_COLORS.items():
        w, f, e = spectra[g]
        ax_full.plot(w, f, color=color, lw=0.5, alpha=0.8, label=g)
        ax_full.fill_between(w, f - e, f + e, color=color, alpha=0.1)

    # Add rest-frame axis on top
    ax_rest = ax_full.twiny()
    x1, x2 = ax_full.get_xlim()
    ax_rest.set_xlim(x1 * 1e4 / (1 + Z), x2 * 1e4 / (1 + Z))
    ax_rest.set_xlabel(r'Rest wavelength ($\AA$)', fontsize=12)

    # Mark lines on full spectrum
    mark_lines(ax_full, BALMER_LINES, Z, 'red', ls='--', fontsize=9, offset_idx=0)
    mark_lines(ax_full, PASCHEN_LINES, Z, 'blue', ls='-.', fontsize=9, offset_idx=1)
    mark_lines(ax_full, HELIUM_LINES, Z, 'green', ls=':', fontsize=9, offset_idx=2)
    mark_lines(ax_full, FORBIDDEN_LINES, Z, 'gray', ls=':', fontsize=8, alpha=0.4,
               offset_idx=0)

    ax_full.set_xlabel(r'Observed wavelength ($\mu$m)', fontsize=14)
    ax_full.set_ylabel(r'Flux ($\mu$Jy)', fontsize=14)
    ax_full.set_title(
        f'bluejay-south-v4-12020  z={Z:.3f}  '
        f'(log N(n=2)={LOG_NHI:.2f}, b={B_ABS:.0f} km/s, '
        f'$\\Delta$WAIC={DWAIC:.0f})',
        fontsize=13,
    )
    ax_full.legend(loc='upper right', fontsize=11)
    ax_full.minorticks_on()

    # ---- Zoomed panels ----
    # Define zoom windows: (center_um, half_width_um, title, lines_dict)
    zoom_panels = [
        (ax_pad,
         (9650.0 * (1 + Z) / 1e4, 0.18),
         r'Pa$\delta$ + Pa$\epsilon$ + Pa 9',
         {r'Pa$\delta$': 10052.1, r'Pa$\epsilon$': 9546.0, r'Pa 9': 9229.0}),
        (ax_pag,
         (10941.1 * (1 + Z) / 1e4, 0.12),
         r'Pa$\gamma$ + HeI 10833',
         {r'Pa$\gamma$': 10941.1, r'HeI 10833': 10833.0}),
        (ax_pab,
         (12821.6 * (1 + Z) / 1e4, 0.15),
         r'Pa$\beta$',
         {r'Pa$\beta$': 12821.6}),
    ]

    for ax, (center, hw), title, lines in zoom_panels:
        xmin, xmax = center - hw, center + hw
        ax.set_xlim(xmin, xmax)

        # Plot G395M data (covers all these features)
        w, f, e = spectra['G395M']
        mask = (w >= xmin - 0.02) & (w <= xmax + 0.02)
        if mask.sum() > 0:
            ax.step(w[mask], f[mask], where='mid', color='C2', lw=1, zorder=3)
            ax.fill_between(w[mask], f[mask] - e[mask], f[mask] + e[mask],
                            step='mid', color='C2', alpha=0.2, zorder=2)

        # Auto y-limits from visible data
        vis = (w >= xmin) & (w <= xmax)
        if vis.sum() > 0:
            fvis = f[vis]
            evis = e[vis]
            ylo = np.nanpercentile(fvis - evis, 5)
            yhi = np.nanpercentile(fvis + evis, 95)
            margin = 0.15 * (yhi - ylo)
            ax.set_ylim(ylo - margin, yhi + 2 * margin)

        # Mark lines (behind spectrum)
        for name, lam_rest in lines.items():
            lam_obs = lam_rest * (1 + Z) / 1e4
            color = 'green' if 'HeI' in name else 'blue'
            ax.axvline(lam_obs, ls='--', color=color, alpha=0.25, lw=1, zorder=1)
            ax.text(lam_obs, ax.get_ylim()[1] * 0.97, name,
                    ha='center', va='top', fontsize=10, color=color,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec=color, alpha=0.8), zorder=5)

        # Add rest-frame axis
        ax_r = ax.twiny()
        x1, x2 = ax.get_xlim()
        ax_r.set_xlim(x1 * 1e4 / (1 + Z), x2 * 1e4 / (1 + Z))
        ax_r.set_xlabel(r'Rest $\lambda$ ($\AA$)', fontsize=10)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(r'Observed $\lambda$ ($\mu$m)', fontsize=11)
        ax.set_ylabel(r'Flux ($\mu$Jy)', fontsize=11)
        ax.minorticks_on()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, 'bluejay-south-v4-12020_paschen_absorption.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()
