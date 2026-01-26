#!/usr/bin/env python
"""
Extract line fluxes from UNITE summary files and create diagnostic plots.

This script:
1. Reads all *_summary.csv files from the UNITE output directory
2. Extracts total, narrow, and broad fluxes for Halpha, Hbeta, and [OIII]
3. Creates diagnostic plots:
   - Total Ha vs Hb with Case B recombination lines
   - Sum of Ha+Hb vs [OIII]
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# Configuration
OUTPUT_DIRECTORY = 'out_all_rubies'
SPEC_TABLE_PATH = '/Users/jiamuh/python/cloudy_lrds/cloudy_lrds/dja_spec/rubies_all_spectra/rubies_all_spectra_table.fits'

# Cosmology for luminosity distance calculation
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def case_b_ratio(temperature):
    """
    Calculate Case B Ha/Hb ratio for a given temperature.

    Uses the fit from Osterbrock & Ferland (2006), Table 4.4
    For temperatures from 5000K to 20000K

    Parameters
    ----------
    temperature : float or array
        Temperature in Kelvin

    Returns
    -------
    ratio : float or array
        Ha/Hb flux ratio
    """
    # Approximate values from Osterbrock & Ferland (2006)
    # Using a simple polynomial fit to the tabulated values
    T4 = temperature / 1e4  # Temperature in units of 10^4 K

    # Simple approximation (valid for 5000K - 20000K)
    ratio = 2.86 + 0.17 * (1 - T4)

    return ratio


def extract_flux_data(summary_file, spec_table):
    """
    Extract flux data from a summary CSV file.

    Parameters
    ----------
    summary_file : str
        Path to the summary CSV file
    spec_table : astropy.table.Table
        Table with source information including redshifts

    Returns
    -------
    data : dict
        Dictionary with extracted flux measurements
    """
    # Read summary file
    df = pd.read_csv(summary_file, index_col=0)

    # Extract source ID from filename
    # Format: {root}-{srcid}_{config}_summary.csv
    basename = os.path.basename(summary_file)
    parts = basename.replace('_summary.csv', '').split('-')
    srcid = int(parts[-1].split('_')[0])

    # Get redshift from spec_table
    source_rows = spec_table[spec_table['srcid'] == srcid]
    if len(source_rows) == 0:
        print(f"Warning: srcid {srcid} not found in spec_table")
        return None

    # Handle different redshift column names
    if 'z' in source_rows.colnames:
        z = float(source_rows[0]['z'])
    elif 'z_spec' in source_rows.colnames:
        z = float(source_rows[0]['z_spec'])
        if not np.isfinite(z) and 'z_map' in source_rows.colnames:
            z = float(source_rows[0]['z_map'])
    elif 'z_map' in source_rows.colnames:
        z = float(source_rows[0]['z_map'])
    else:
        print(f"Warning: No redshift column found for srcid {srcid}")
        return None

    # Calculate luminosity distance in cm
    D_L = cosmo.luminosity_distance(z).to(u.cm).value

    # Conversion factor from flux (erg/s/cm^2) to luminosity (erg/s)
    # L = 4 * pi * D_L^2 * F
    flux_to_lum = 4 * np.pi * D_L**2

    data = {
        'srcid': srcid,
        'z': z,
        'D_L': D_L,
    }

    # Extract fluxes for each line type
    line_names = [
        ('Ha', 'HI_6564.61'),
        ('Hb', 'HI_4862.68'),
        ('OIII', 'OIII_5008.24'),
    ]

    for short_name, line_id in line_names:
        for component in ['total', 'narrow', 'broad']:
            flux_key = f"{line_id}_{component}_flux"

            if flux_key in df.index:
                # Extract P16, P50, P84
                p16 = df.loc[flux_key, 'P16']
                p50 = df.loc[flux_key, 'P50']
                p84 = df.loc[flux_key, 'P84']

                # Store flux values (in 10^-20 erg/s/cm^2 as per unite/fitting.py:412)
                data[f'{short_name}_{component}_flux'] = p50
                data[f'{short_name}_{component}_flux_err_low'] = p50 - p16
                data[f'{short_name}_{component}_flux_err_high'] = p84 - p50

                # Convert to luminosity (erg/s)
                lum = p50 * 1e-20 * flux_to_lum
                lum_err_low = (p50 - p16) * 1e-20 * flux_to_lum
                lum_err_high = (p84 - p50) * 1e-20 * flux_to_lum

                # Store in log10(L) for plotting
                if lum > 0:
                    data[f'{short_name}_{component}_logL'] = np.log10(lum)
                    # Error propagation for log
                    data[f'{short_name}_{component}_logL_err_low'] = np.log10(lum) - np.log10(lum - lum_err_low) if lum > lum_err_low else 0.3
                    data[f'{short_name}_{component}_logL_err_high'] = np.log10(lum + lum_err_high) - np.log10(lum)
                else:
                    data[f'{short_name}_{component}_logL'] = np.nan
                    data[f'{short_name}_{component}_logL_err_low'] = np.nan
                    data[f'{short_name}_{component}_logL_err_high'] = np.nan

    return data


def create_diagnostic_plots(data_table, output_dir='plots'):
    """
    Create diagnostic plots from extracted line flux data.

    Parameters
    ----------
    data_table : astropy.table.Table
        Table with extracted flux and luminosity data
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set up plotting style
    plt.rcParams.update({
        'text.usetex': True,
        'axes.linewidth': 2,
        'font.family': 'serif',
        'font.weight': 'heavy',
        'font.size': 20,
    })
    plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \boldmath'

    # ========================================================================
    # Plot 1: Total Ha vs Hb with Case B lines
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 10))

    # Filter out invalid data for total
    valid_total = (
        np.isfinite(data_table['Ha_total_logL']) &
        np.isfinite(data_table['Hb_total_logL'])
    )

    # Filter out invalid data for narrow
    valid_narrow = (
        np.isfinite(data_table['Ha_narrow_logL']) &
        np.isfinite(data_table['Hb_narrow_logL'])
    )

    if np.sum(valid_total) > 0:
        # Plot total (narrow + broad) with black
        ax.errorbar(
            data_table['Hb_total_logL'][valid_total],
            data_table['Ha_total_logL'][valid_total],
            xerr=[data_table['Hb_total_logL_err_low'][valid_total],
                  data_table['Hb_total_logL_err_high'][valid_total]],
            yerr=[data_table['Ha_total_logL_err_low'][valid_total],
                  data_table['Ha_total_logL_err_high'][valid_total]],
            fmt='o',
            markersize=8,
            capsize=3,
            label=r'$\rm Total~(narrow + broad)$',
            color='black',
            alpha=0.8,
            zorder=3
        )

        # Plot narrow component with lighter black
        if np.sum(valid_narrow) > 0:
            ax.errorbar(
                data_table['Hb_narrow_logL'][valid_narrow],
                data_table['Ha_narrow_logL'][valid_narrow],
                xerr=[data_table['Hb_narrow_logL_err_low'][valid_narrow],
                      data_table['Hb_narrow_logL_err_high'][valid_narrow]],
                yerr=[data_table['Ha_narrow_logL_err_low'][valid_narrow],
                      data_table['Ha_narrow_logL_err_high'][valid_narrow]],
                fmt='s',
                markersize=7,
                capsize=3,
                label=r'$\rm Narrow$',
                color='gray',
                alpha=0.7,
                zorder=2
            )

        # Get the range for Case B lines
        x_min, x_max = ax.get_xlim()
        x_range = np.linspace(x_min, x_max, 100)

        # Plot Case B recombination lines for different temperatures
        # Change max T to 2e5K
        temperatures = np.logspace(3, np.log10(2e5), 15)  # 10^3 to 2e5 K
        colors = plt.cm.plasma(np.linspace(0, 1, len(temperatures)))

        # Plot all Case B lines with consistent thickness
        for temp, color in zip(temperatures, colors):
            ratio = case_b_ratio(temp)
            y_range = x_range + np.log10(ratio)
            ax.plot(x_range, y_range, '--', color=color, alpha=0.7,
                   linewidth=1.5, zorder=1)

        # Add colorbar for temperature
        sm = plt.cm.ScalarMappable(
            cmap='plasma',
            norm=plt.Normalize(vmin=np.log10(temperatures.min()),
                              vmax=np.log10(temperatures.max()))
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=r'$\log(T~[\rm K])$', pad=0.02)
        cbar.ax.tick_params(length=6, width=1.5, direction='in')

        ax.set_xlabel(r'$\log~L_{\rm H\beta}~[\rm erg~s^{-1}]$', fontsize=22)
        ax.set_ylabel(r'$\log~L_{\rm H\alpha}~[\rm erg~s^{-1}]$', fontsize=22)
        ax.legend(loc='upper left', fontsize=16, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle=':')
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, axis='both', which='major',
                      length=8, width=1.5, direction='in', pad=5)
        ax.tick_params(top=True, right=True, axis='both', which='minor',
                      length=4, width=1, direction='in')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Ha_vs_Hb_total.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved Ha vs Hb plot")
    else:
        print("  ✗ No valid data for Ha vs Hb plot")

    # ========================================================================
    # Plot 2: (Ha + Hb) vs [OIII]
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate Ha + Hb in linear space, then convert to log
    valid = (
        np.isfinite(data_table['Ha_total_logL']) &
        np.isfinite(data_table['Hb_total_logL']) &
        np.isfinite(data_table['OIII_total_logL'])
    )

    if np.sum(valid) > 0:
        # Convert back to linear, sum, then to log
        Ha_lum = 10**data_table['Ha_total_logL'][valid]
        Hb_lum = 10**data_table['Hb_total_logL'][valid]
        HaHb_sum = Ha_lum + Hb_lum
        HaHb_logL = np.log10(HaHb_sum)

        # Error propagation (simplified - using quadrature of log errors)
        Ha_err_low = data_table['Ha_total_logL_err_low'][valid]
        Ha_err_high = data_table['Ha_total_logL_err_high'][valid]
        Hb_err_low = data_table['Hb_total_logL_err_low'][valid]
        Hb_err_high = data_table['Hb_total_logL_err_high'][valid]

        # Approximate combined error (conservative estimate)
        HaHb_err_low = np.sqrt(Ha_err_low**2 + Hb_err_low**2)
        HaHb_err_high = np.sqrt(Ha_err_high**2 + Hb_err_high**2)

        ax.errorbar(
            data_table['OIII_total_logL'][valid],
            HaHb_logL,
            xerr=[data_table['OIII_total_logL_err_low'][valid],
                  data_table['OIII_total_logL_err_high'][valid]],
            yerr=[HaHb_err_low, HaHb_err_high],
            fmt='o',
            markersize=8,
            capsize=3,
            label=r'$\rm Total~(narrow + broad)$',
            color='black',
            alpha=0.8
        )

        ax.set_xlabel(r'$\log~L_{[\rm OIII]}~[\rm erg~s^{-1}]$', fontsize=22)
        ax.set_ylabel(r'$\log~L_{\rm H\alpha + H\beta}~[\rm erg~s^{-1}]$', fontsize=22)
        ax.legend(loc='upper left', fontsize=16, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle=':')
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, axis='both', which='major',
                      length=8, width=1.5, direction='in', pad=5)
        ax.tick_params(top=True, right=True, axis='both', which='minor',
                      length=4, width=1, direction='in')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'HaHb_vs_OIII.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved (Ha+Hb) vs [OIII] plot")
    else:
        print("  ✗ No valid data for (Ha+Hb) vs [OIII] plot")


def main():
    """Main function."""

    print("="*70)
    print("Analyzing UNITE Line Flux Results")
    print("="*70)

    # Load spec table for redshifts
    print(f"\nLoading spec table from:\n  {SPEC_TABLE_PATH}")
    spec_table = Table.read(SPEC_TABLE_PATH)

    # Find all summary files
    summary_pattern = os.path.join(OUTPUT_DIRECTORY, 'Results', '*_summary.csv')
    summary_files = sorted(glob.glob(summary_pattern))

    print(f"\nFound {len(summary_files)} summary files")

    if len(summary_files) == 0:
        print("No summary files found. Run run_unite_fits.py first!")
        return

    # Extract data from all files
    all_data = []
    print("\nExtracting flux data...")
    for summary_file in summary_files:
        basename = os.path.basename(summary_file)
        print(f"  Processing: {basename}")

        data = extract_flux_data(summary_file, spec_table)
        if data is not None:
            all_data.append(data)

    if len(all_data) == 0:
        print("No valid data extracted!")
        return

    # Convert to astropy table
    data_table = Table(rows=all_data)

    # Save extracted data
    output_file = os.path.join(OUTPUT_DIRECTORY, 'extracted_line_fluxes.fits')
    data_table.write(output_file, format='fits', overwrite=True)
    print(f"\n✓ Saved extracted data to:\n  {output_file}")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print(f"Total sources: {len(data_table)}")
    print(f"Redshift range: {data_table['z'].min():.2f} - {data_table['z'].max():.2f}")

    # Create diagnostic plots
    print(f"\n{'='*70}")
    print("Creating Diagnostic Plots")
    print(f"{'='*70}")
    plot_dir = os.path.join(OUTPUT_DIRECTORY, 'DiagnosticPlots')
    create_diagnostic_plots(data_table, output_dir=plot_dir)

    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"Plots saved to: {plot_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
