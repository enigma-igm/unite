#!/usr/bin/env python
"""
Plot rest-frame UV spectra (MgII, CIV regions) for all broad-Ha BLAGN
that have medium/high-resolution grating coverage.

For each source with FWHM_Ha >= 1000 km/s and logL_Ha >= 41.7,
checks if a medium-res grating (G140M, G235M, G235H, etc.) covers
the MgII 2800A or CIV 1549A region, and plots the rest-frame spectrum.
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
MASTER_TABLE = os.path.join(DATA_DIR, 'out_dja_absorption', 'master_table.fits')
SPEC_TABLE = os.path.join(DATA_DIR, 'dja_all_spectra', 'dja_all_spectra_table.fits')

FWHM_THRESHOLD = 1000.0   # km/s
LOG_LHA_MIN_BROAD = 41.7

# UV lines to mark (rest-frame Angstrom)
UV_LINES = {
    r'Ly$\alpha$': 1216.0,
    r'NV': 1240.0,
    r'CIV': 1549.0,
    r'HeII': 1640.0,
    r'CIII]': 1909.0,
    r'MgII': 2800.0,
}

# f_nu to f_lambda conversion
C_AA_S = 2.99792458e18
FLAM_UNIT = 1e-20


def fnu_to_flam(wave_um, fnu_ujy):
    """Convert f_nu (uJy) to f_lambda (1e-20 erg/s/cm2/A)."""
    wave_aa = wave_um * 1e4
    return fnu_ujy * 1e-29 * C_AA_S / wave_aa**2 / FLAM_UNIT


def load_spectrum(filepath):
    """Load SPEC1D, return (wave_um, flam, flam_err)."""
    spec = Table.read(filepath, 'SPEC1D')
    w = np.array(spec['wave'])
    f = np.array(spec['flux'])
    e = np.array(spec['err'])
    good = np.isfinite(f) & np.isfinite(e) & (e > 0)
    w, f, e = w[good], f[good], e[good]
    flam = fnu_to_flam(w, f)
    flam_err = fnu_to_flam(w, e)
    return w, flam, flam_err


# =============================================================================
# Main
# =============================================================================

def main():
    # ---- Load tables ----
    print('Loading master table...')
    mt = Table.read(MASTER_TABLE)
    df = mt.to_pandas()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.decode('utf-8').str.strip()

    print('Loading DJA spec table...')
    spec = Table.read(SPEC_TABLE)
    sdf = spec.to_pandas()
    for col in sdf.select_dtypes(include='object').columns:
        sdf[col] = sdf[col].str.decode('utf-8').str.strip()

    # ---- Select broad-Ha sources ----
    broad = df[(df['Ha_total_FWHM'] >= FWHM_THRESHOLD) &
               (df['Ha_total_logL'] >= LOG_LHA_MIN_BROAD) &
               (df['catalog'] == 'DJA')].copy()
    print(f'Broad-Ha DJA sources: {len(broad)}')

    # ---- Build grating lookup: (root, srcid) -> list of (grating, file, wmin, wmax) ----
    grating_lookup = {}
    for _, s in sdf.iterrows():
        g = s['grating']
        if 'PRISM' in g.upper():
            continue
        key = (s['root'], int(s['srcid']))
        if key not in grating_lookup:
            grating_lookup[key] = []
        grating_lookup[key].append({
            'grating': g,
            'file': s['filename'] if 'filename' in sdf.columns else s['file'],
            'wmin': s['wmin'],
            'wmax': s['wmax'],
        })

    # ---- For each line region, find sources with grating coverage ----
    regions = {
        'MgII': {'center': 2800.0, 'window': (2600, 3100), 'lines': ['MgII', 'CIII]']},
        'CIV': {'center': 1549.0, 'window': (1350, 1750), 'lines': ['CIV', 'HeII', r'Ly$\alpha$', 'NV']},
    }

    plt.rcParams.update({
        'text.usetex': True,
        'axes.linewidth': 1.5,
        'font.family': 'serif',
        'font.weight': 'heavy',
        'font.size': 11,
    })
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for region_name, rinfo in regions.items():
        lam_center = rinfo['center']
        rest_lo, rest_hi = rinfo['window']
        line_names = rinfo['lines']

        # Find sources with grating coverage at this line
        sources = []
        for _, row in broad.iterrows():
            z = row['z']
            key = (row['root'], int(row['srcid']))
            if key not in grating_lookup:
                continue

            obs_center = lam_center * (1 + z) / 1e4  # microns

            # Find best grating covering this line
            best = None
            for gi in grating_lookup[key]:
                if gi['wmin'] <= obs_center <= gi['wmax']:
                    # Prefer higher resolution (H > M)
                    if best is None or 'H' in gi['grating']:
                        best = gi
            if best is None:
                continue

            sources.append({
                'root': row['root'],
                'srcid': int(row['srcid']),
                'z': z,
                'logL_Ha': row['Ha_total_logL'],
                'FWHM_Ha': row['Ha_total_FWHM'],
                'is_lrd': bool(row['is_lrd']),
                'grating': best['grating'],
                'file': best['file'],
            })

        if not sources:
            print(f'No sources with grating coverage at {region_name}')
            continue

        # Sort by redshift
        sources = sorted(sources, key=lambda x: x['z'])
        N = len(sources)
        print(f'\n{region_name}: {N} sources with grating coverage')

        # ---- Grid plot ----
        ncols = 6
        nrows = int(np.ceil(N / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)

        for idx, src in enumerate(sources):
            ax = axes[idx // ncols, idx % ncols]
            z = src['z']

            # Load spectrum
            fpath = os.path.join(SPEC_DIR, src['file'])
            if not os.path.exists(fpath):
                ax.text(0.5, 0.5, 'file missing', transform=ax.transAxes,
                        ha='center', fontsize=9)
                continue
            try:
                w_um, flam, flam_err = load_spectrum(fpath)
            except Exception as e:
                ax.text(0.5, 0.5, f'load err', transform=ax.transAxes,
                        ha='center', fontsize=9)
                continue

            # Convert to rest-frame
            w_rest = w_um * 1e4 / (1 + z)  # Angstrom

            # Select window
            mask = (w_rest >= rest_lo) & (w_rest <= rest_hi)
            if mask.sum() < 5:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', fontsize=9)
                continue

            wr = w_rest[mask]
            fr = flam[mask]
            er = flam_err[mask]

            # Plot
            ax.step(wr, fr, where='mid', color='k', lw=0.7, zorder=3)
            ax.fill_between(wr, fr - er, fr + er, step='mid',
                            color='gray', alpha=0.2, zorder=2)

            # Auto y-limits
            ylo = np.nanpercentile(fr - er, 2)
            yhi = np.nanpercentile(fr + er, 98)
            if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
                margin = 0.15 * (yhi - ylo)
                ax.set_ylim(ylo - margin, yhi + margin)

            # Mark UV lines
            for lname in line_names:
                lam = UV_LINES.get(lname)
                if lam and rest_lo <= lam <= rest_hi:
                    ax.axvline(lam, ls='--', color='red', alpha=0.3, lw=0.8,
                               zorder=1)
                    ax.text(lam, ax.get_ylim()[1] * 0.97, lname,
                            ha='center', va='top', fontsize=7, color='red')

            # Title
            lrd_tag = ' LRD' if src['is_lrd'] else ''
            ax.set_title(
                rf"$z={z:.2f}$  $\rm FWHM={src['FWHM_Ha']:.0f}${lrd_tag}",
                fontsize=8, pad=2,
            )

            # Source name below title
            name = f"{src['root']}-{src['srcid']}"
            # Truncate long names
            if len(name) > 28:
                name = name[:12] + '..' + name[-12:]
            ax.text(0.02, 0.95, name, transform=ax.transAxes, fontsize=5.5,
                    va='top', color='0.4')

            # Grating label
            ax.text(0.98, 0.95, src['grating'], transform=ax.transAxes,
                    fontsize=6, va='top', ha='right', color='C0')

            ax.set_xlim(rest_lo, rest_hi)
            ax.minorticks_on()
            ax.tick_params(which='major', length=4, width=1, direction='in',
                           top=True, right=True)
            ax.tick_params(which='minor', length=2, width=0.5, direction='in',
                           top=True, right=True)

        # Labels on edge panels only
        for i in range(nrows):
            axes[i, 0].set_ylabel(
                r'$f_\lambda$ $[\rm 10^{-20}~erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]$',
                fontsize=8)
        for j in range(ncols):
            axes[-1, j].set_xlabel(r'$\rm Rest~\lambda~[\AA]$', fontsize=9)

        # Hide empty panels
        for idx in range(N, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle(
            rf'$\rm Rest\text{{-}}frame~{region_name}~region~for~broad\text{{-}}H\alpha~BLAGN$'
            rf'$\rm ~(FWHM \geq {FWHM_THRESHOLD:.0f}~km/s,~N={N})$',
            fontsize=16, y=1.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        outpath = os.path.join(OUTPUT_DIR, f'UV_{region_name}_blagn_grid.png')
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {outpath}')

    print('\nDone.')


if __name__ == '__main__':
    main()
