#!/usr/bin/env python
"""
Plot the full spectrum with the best-fit absorption model overlaid.
Shows the complete wavelength range (not just continuum regions).
"""

import os
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
from collections import defaultdict

import jax.numpy as jnp
from unite.spectra import NIRSpecSpectra
from unite.absorption import full_balmer_transmission, BALMER_LIMIT


def load_and_plot(srcid, data_dir, output_dir, label='broad_abs'):
    """Load spectra + fit results and plot full spectrum with absorption."""

    SPEC_TABLE = os.path.join(data_dir, 'rubies_all_spectra/rubies_all_spectra_table.fits')
    SPEC_DIR = os.path.join(data_dir, 'rubies_all_spectra')

    # Load and prep table
    spec_table = Table.read(SPEC_TABLE)
    spec_table.rename_column('filename', 'file')
    z_col = np.where(
        (np.isfinite(spec_table['z_spec'])) & (spec_table['z_spec'] > 0),
        spec_table['z_spec'], spec_table['z_map']
    )
    spec_table['z'] = z_col
    spec_table['zfit'] = spec_table['z']
    spec_table['grating'] = [g.split('_')[0].upper() for g in spec_table['grating']]

    rows = spec_table[spec_table['srcid'] == srcid]

    # Filter best per grating
    grating_rows = defaultdict(list)
    for i, row in enumerate(rows):
        grating_rows[row['grating']].append(i)
    best = []
    for g, idxs in grating_rows.items():
        if len(idxs) == 1:
            best.append(idxs[0])
        else:
            grades = [float(rows[i]['grade']) for i in idxs]
            best.append(idxs[np.argmax(grades)])
    rows = rows[sorted(best)]

    root = rows[0]['root']

    # Load spectra
    spectra = NIRSpecSpectra(rows, SPEC_DIR)
    z = spectra.redshift_initial
    um_to_aa = spectra.spectra[0].λ_unit.to('Angstrom')

    # Load fit samples
    npz_path = os.path.join(output_dir, 'Results', f'{root}-{srcid}_{label}_full.npz')
    samples = dict(np.load(npz_path, allow_pickle=True))

    # Get absorption params (median)
    has_abs = 'log_NHI' in samples
    if has_abs:
        logN = np.median(samples['log_NHI'])
        b_abs = np.median(samples['b_abs'])
        dv_abs = np.median(samples['delta_v_abs'])
        logN_16, logN_84 = np.quantile(samples['log_NHI'], [0.16, 0.84])
        print(f'Absorption params: logN={logN:.2f} [{logN_16:.2f}, {logN_84:.2f}], '
              f'b={b_abs:.0f} km/s, dv={dv_abs:.0f} km/s')

    # Key Balmer line positions (rest frame, Angstrom -> micron)
    balmer_marks = [
        (6564.61 / 1e4, r'H$\alpha$'),
        (4862.68 / 1e4, r'H$\beta$'),
        (4341.68 / 1e4, r'H$\gamma$'),
        (4102.89 / 1e4, r'H$\delta$'),
        (3970.07 / 1e4, r'H$\epsilon$'),
        (BALMER_LIMIT / 1e4, 'BL'),  # Balmer limit
    ]

    n_spectra = len(spectra.spectra)
    fig, axes = plt.subplots(n_spectra, 1, figsize=(14, max(5, 4 * n_spectra)),
                             squeeze=False)
    axes = axes[:, 0]  # squeeze column dim only

    for i, spectrum in enumerate(spectra.spectra):
        ax = axes[i]
        _, wave, _, flux, err = spectrum()
        wave = np.array(wave)
        flux = np.array(flux)
        err = np.array(err)

        # Plot data
        ax.step(wave, flux, where='mid', color='k', lw=0.8, label='Data', zorder=5)
        ax.fill_between(wave, flux - err, flux + err, step='mid',
                        color='k', alpha=0.1, zorder=1)

        # Model is on the restricted wavelength grid (continuum regions only)
        wave_model_key = f'{spectrum.name}_wavelength'
        model_key = f'{spectrum.name}_model'
        if model_key in samples and wave_model_key in samples:
            wave_model = samples[wave_model_key]
            model = samples[model_key]
            best_idx = int(samples['logP'].argmax())

            # Thin sample traces
            n_plot = min(50, model.shape[0])
            for k in range(0, model.shape[0], max(1, model.shape[0] // n_plot)):
                ax.step(wave_model, model[k], where='mid', color='#E20134',
                        alpha=0.05, zorder=2)

            # Best fit
            ax.step(wave_model, model[best_idx], where='mid', color='#A40122',
                    lw=1.5, label='Best model', zorder=6)

        # Continuum
        cont_key = f'{spectrum.name}_cont'
        if cont_key in samples and wave_model_key in samples:
            cont = samples[cont_key]
            flux_scale_key = f'{spectrum.name}_flux'
            fs = samples[flux_scale_key][best_idx] if flux_scale_key in samples else 1.0
            ax.step(wave_model, fs * cont[best_idx], where='mid', color='gray',
                    lw=1, ls='--', label='Continuum', zorder=4)

        # Plot full transmission (all Balmer lines + b-f) on twin axis
        if has_abs:
            wave_aa = jnp.array(wave) * um_to_aa
            T_med = np.array(full_balmer_transmission(
                wave_aa, z, logN, b_abs, dv_abs, n_max=50))

            ax2 = ax.twinx()
            ax2.plot(wave, T_med, color='purple', lw=1.5, alpha=0.8,
                     label='Transmission')
            ax2.set_ylim(-0.05, 1.15)
            ax2.set_ylabel('T(λ)', color='purple', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='purple')

        ax.set_ylabel(f'{spectrum.name}\n' + r'$f_\lambda$', fontsize=10)
        ax.set_xlim(wave.min(), wave.max())
        # Auto-scale y to clip outliers (use 1st-99th percentile of flux)
        valid = np.isfinite(flux)
        if valid.any():
            lo, hi = np.percentile(flux[valid], [1, 99])
            margin = (hi - lo) * 0.3
            ax.set_ylim(lo - margin, hi + margin)

        # Mark key Balmer lines (after ylim is set so text stays inside axes)
        for lam_rest, name in balmer_marks:
            lam_obs = lam_rest * (1 + z)
            if wave.min() < lam_obs < wave.max():
                ax.axvline(lam_obs, color='blue', ls=':', alpha=0.5, zorder=0)
                ax.text(lam_obs, ax.get_ylim()[1] * 0.95, name,
                        ha='center', fontsize=8, color='blue')

        ax.legend(loc='upper left', fontsize=8)

    axes[-1].set_xlabel('Observed wavelength (μm)', fontsize=11)

    # Title
    title = f'{root}-{srcid}  z={z:.3f}'
    if has_abs:
        title += (f'  |  log N_HI={logN:.2f}, '
                  f'b={b_abs:.0f} km/s, Δv={dv_abs:.0f} km/s')
    fig.suptitle(title, fontsize=12, y=1.02)

    # Rest frame axis on top subplot
    ax_top = axes[0]
    rest_ax = ax_top.secondary_xaxis(
        'top',
        functions=(lambda x: x / (1 + z), lambda x: x * (1 + z)),
    )
    rest_ax.set_xlabel('Rest wavelength (μm)', fontsize=10)

    fig.subplots_adjust(hspace=0.35, top=0.90, bottom=0.08, left=0.08, right=0.92)
    outpath = os.path.join(output_dir, 'Plots', f'{root}-{srcid}_{label}_full.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    DATA_DIR = '/Users/jiamuh/python/unite_fitting_data'

    # Plot 49140
    print('=== RUBIES-EGS-49140 ===')
    load_and_plot(49140, DATA_DIR,
                  os.path.join(DATA_DIR, 'test_absorption_49140'))

    # Plot 55604 (if results exist)
    out_55604 = os.path.join(DATA_DIR, 'test_absorption_55604')
    if os.path.exists(os.path.join(out_55604, 'Results')):
        print('\n=== RUBIES-EGS-55604 ===')
        load_and_plot(55604, DATA_DIR, out_55604)
    else:
        print('\n55604 results not ready yet')
