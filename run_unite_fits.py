#!/usr/bin/env python
"""
Run UNITE fitting on all spectra from the broadline_spec_table.fits

This script loops over all spectra downloaded in rubies_broadline_spectra
and runs the UNITE fitting pipeline on each unique source.
"""

import os
import json
import numpy as np
from astropy.table import Table
from unite.spectra import NIRSpecSpectra
from unite.fitting import NIRSpecFit
from multiprocessing import Pool, cpu_count
from functools import partial

# Configure JAX for parallel processing
# Set XLA flags to use single thread per process to avoid conflicts
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_ENABLE_X64'] = '1'  # Enable 64-bit precision if needed

# Configuration
SPEC_TABLE_PATH = '/Users/jiamuh/python/cloudy_lrds/cloudy_lrds/dja_spec/rubies_all_spectra/rubies_all_spectra_table.fits'
SPECTRA_DIRECTORY = '/Users/jiamuh/python/cloudy_lrds/cloudy_lrds/dja_spec/rubies_all_spectra'
CONFIG_FILE = 'example-config-broad.json'
OUTPUT_DIRECTORY = 'out_all_rubies'

# MCMC parameters
N_SAMPLES = 1000
N_WARMUP = 500

# Filter options
SKIP_NEGATIVE_SRCIDS = True  # Set to False to include negative srcids

# Parallel processing options
USE_PARALLEL = True  # Set to False to run sequentially
N_CORES = max(1, cpu_count() - 2)  # Use all cores minus 2 to prevent crashes


def process_single_source(srcid, spec_table, config, spectra_directory, output_directory,
                         n_samples, n_warmup):
    """
    Process a single source. This function is designed to be called in parallel.

    Parameters
    ----------
    srcid : int
        Source ID to process
    spec_table : astropy.table.Table
        Full spectrum table
    config : dict
        Configuration dictionary
    spectra_directory : str
        Directory containing spectra files
    output_directory : str
        Directory for output files
    n_samples : int
        Number of MCMC samples
    n_warmup : int
        Number of warmup samples

    Returns
    -------
    tuple
        (srcid, success, message)
    """
    try:
        # Get all rows for this source
        source_rows = spec_table[spec_table['srcid'] == srcid]

        if len(source_rows) == 0:
            return (srcid, False, "No data found")

        # Check if output already exists
        root = source_rows[0]['root']
        output_files = [
            os.path.join(output_directory, 'Results', f"{root}-{srcid}_{config['Name']}_summary.csv"),
            os.path.join(output_directory, 'Plots', f"{root}-{srcid}_{config['Name']}_fit.png")
        ]

        if all(os.path.exists(f) for f in output_files):
            return (srcid, True, "Already processed")

        # Run the fit
        NIRSpecFit(
            config,
            source_rows,
            spectra_directory=spectra_directory,
            output_directory=output_directory,
            N=n_samples,
            num_warmup=n_warmup,
            verbose=False  # Suppress output for parallel processing
        )

        return (srcid, True, "Success")

    except ValueError as e:
        if "No Line Coverage" in str(e):
            return (srcid, False, "No line coverage (lines outside spectral range)")
        else:
            return (srcid, False, f"ValueError: {e}")
    except IndexError as e:
        return (srcid, False, f"IndexError: {e} (no valid data or lines outside coverage)")
    except Exception as e:
        return (srcid, False, f"Error: {e}")


def main():
    """Main function to run UNITE fits on all sources."""

    print("="*70)
    print("UNITE Fitting for RUBIES All Sources")
    print("="*70)

    # Load the spectrum table
    print(f"\nLoading spectrum table from:\n  {SPEC_TABLE_PATH}")
    spec_table = Table.read(SPEC_TABLE_PATH)
    print(f"  Loaded {len(spec_table)} spectra")

    # Handle different column names for different table formats
    if 'filename' in spec_table.colnames and 'file' not in spec_table.colnames:
        spec_table.rename_column('filename', 'file')
        print("  Renamed 'filename' column to 'file'")

    # Handle redshift: prefer z_spec, fallback to z_map
    if 'z' not in spec_table.colnames:
        if 'z_spec' in spec_table.colnames:
            # Use z_spec if available, fallback to z_map if z_spec is NaN
            if 'z_map' in spec_table.colnames:
                z_col = np.where(np.isfinite(spec_table['z_spec']),
                                spec_table['z_spec'],
                                spec_table['z_map'])
            else:
                z_col = spec_table['z_spec']
            spec_table['z'] = z_col
            print("  Created 'z' column from z_spec (with z_map fallback)")
        elif 'z_map' in spec_table.colnames:
            spec_table['z'] = spec_table['z_map']
            print("  Created 'z' column from z_map")

    # Fix grating names: remove filter suffixes (_F290LP, _CLEAR, etc.)
    # UNITE expects just the grating part (G395M, PRISM) to match resolution files
    if 'grating' in spec_table.colnames:
        original_gratings = spec_table['grating'].copy()
        # Extract just the grating part before underscore (if present)
        spec_table['grating'] = [g.split('_')[0] for g in spec_table['grating']]
        if not np.all(original_gratings == spec_table['grating']):
            print("  Stripped filter suffixes from grating names (e.g., G395M_F290LP -> G395M)")

    # Load configuration
    print(f"\nLoading configuration from:\n  {CONFIG_FILE}")
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    # Get unique source IDs
    unique_srcids = np.unique(spec_table['srcid'])

    # Filter out negative srcids if requested
    if SKIP_NEGATIVE_SRCIDS:
        n_negative = np.sum(unique_srcids < 0)
        if n_negative > 0:
            print(f"\nFiltering out {n_negative} sources with negative srcids")
            unique_srcids = unique_srcids[unique_srcids >= 0]

    print(f"\nFound {len(unique_srcids)} unique sources to process")

    # Create output directory
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Process sources
    if USE_PARALLEL:
        print(f"\n{'='*70}")
        print(f"Running in PARALLEL mode using {N_CORES} cores")
        print(f"{'='*70}")

        # Create partial function with fixed arguments
        process_func = partial(
            process_single_source,
            spec_table=spec_table,
            config=config,
            spectra_directory=SPECTRA_DIRECTORY,
            output_directory=OUTPUT_DIRECTORY,
            n_samples=N_SAMPLES,
            n_warmup=N_WARMUP
        )

        # Process in parallel with progress tracking
        print(f"\nProcessing {len(unique_srcids)} sources...")
        with Pool(processes=N_CORES) as pool:
            # Use imap_unordered for progress tracking
            results = []
            for i, result in enumerate(pool.imap_unordered(process_func, unique_srcids), 1):
                results.append(result)
                srcid, success, message = result
                status = "✓" if success else "✗"
                # Print progress every 10 sources or for failures
                if i % 10 == 0 or not success:
                    print(f"  [{i}/{len(unique_srcids)}] {status} srcid={srcid}: {message}")

        # Print summary
        print(f"\n{'='*70}")
        print("Processing Summary")
        print(f"{'='*70}")

        n_success = 0
        n_skip = 0
        n_fail = 0
        failed_srcids = []

        for srcid, success, message in results:
            if success:
                if message == "Already processed":
                    n_skip += 1
                else:
                    n_success += 1
            else:
                n_fail += 1
                failed_srcids.append((srcid, message))

        print(f"Total sources: {len(unique_srcids)}")
        print(f"  Successfully processed: {n_success}")
        print(f"  Skipped (already done): {n_skip}")
        print(f"  Failed: {n_fail}")

        if failed_srcids and n_fail <= 20:
            print(f"\nFailed sources:")
            for srcid, message in failed_srcids:
                print(f"  ✗ {srcid}: {message}")
        elif n_fail > 20:
            print(f"\nShowing first 20 failed sources:")
            for srcid, message in failed_srcids[:20]:
                print(f"  ✗ {srcid}: {message}")
            print(f"  ... and {n_fail - 20} more")

        print(f"{'='*70}")

    else:
        print(f"\n{'='*70}")
        print("Running in SEQUENTIAL mode")
        print(f"{'='*70}")

        # Loop over each unique source sequentially
        for i, srcid in enumerate(unique_srcids, 1):
            print(f"\n{'='*70}")
            print(f"Processing source {i}/{len(unique_srcids)}: srcid={srcid}")
            print(f"{'='*70}")

            # Get all rows for this source
            source_rows = spec_table[spec_table['srcid'] == srcid]
            print(f"  Found {len(source_rows)} disperser(s) for this source:")
            for row in source_rows:
                print(f"    - {row['grating']}: {row['file']}")

            # Check if output already exists
            root = source_rows[0]['root']
            output_files = [
                os.path.join(OUTPUT_DIRECTORY, 'Results', f"{root}-{srcid}_{config['Name']}_summary.csv"),
                os.path.join(OUTPUT_DIRECTORY, 'Plots', f"{root}-{srcid}_{config['Name']}_fit.png")
            ]

            if all(os.path.exists(f) for f in output_files):
                print(f"  Output already exists for {srcid}, skipping...")
                continue

            try:
                # Run the fit
                print(f"  Running UNITE fit...")
                print(f"    Redshift: {source_rows[0]['z']:.4f}")
                print(f"    N_samples: {N_SAMPLES}")
                print(f"    N_warmup: {N_WARMUP}")

                NIRSpecFit(
                    config,
                    source_rows,
                    spectra_directory=SPECTRA_DIRECTORY,
                    output_directory=OUTPUT_DIRECTORY,
                    N=N_SAMPLES,
                    num_warmup=N_WARMUP
                )

                print(f"  ✓ Successfully completed fit for source {srcid}")

            except ValueError as e:
                if "No Line Coverage" in str(e):
                    print(f"  ✗ Skipping source {srcid}: No line coverage (lines outside spectral range)")
                else:
                    print(f"  ✗ Error fitting source {srcid}: {e}")
                continue
            except IndexError as e:
                print(f"  ✗ Error fitting source {srcid}: {e}")
                print(f"     (This may indicate no valid data or lines outside spectral coverage)")
                continue
            except Exception as e:
                print(f"  ✗ Error fitting source {srcid}: {e}")
                continue

    print(f"\n{'='*70}")
    print("All sources processed!")
    print(f"Results saved to: {OUTPUT_DIRECTORY}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
