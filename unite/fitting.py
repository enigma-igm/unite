"""
Fitting functions for spectral data
"""

# Standard library
import re
import os

# Typing
from typing import Dict, Tuple

# Data Science
import pandas as pd

# Astropy packages
import astropy.units as u
from astropy.io import fits
from astropy.table import Table, hstack

# Numpyro
from numpyro import infer, optim
from numpyro.handlers import trace, seed, substitute

# JAX
import numpy as np
from jax import random, vmap, numpy as jnp

# unite
from unite.model import multiSpecModel
from unite.spectra import NIRSpecSpectra
from unite import utils, initial, parameters

# Plotting packages
from matplotlib import pyplot


def NIRSpecFit(
    config: dict,
    rows: Table,
    spectra_directory: str,
    output_directory: str,
    N: int = 500,
    num_warmup: int = 250,
    backend: str = 'MCMC',
    verbose=True,
) -> None:
    # Get the model arguments
    config, model_args = NIRSpecModelArgs(config, rows, spectra_directory)

    # Get the random key
    rng_key = random.PRNGKey(0)

    # Fit the data
    match backend:
        case 'MCMC':
            samples, extras = MCMCFit(
                model_args, rng_key, N=N, num_warmup=num_warmup, verbose=verbose
            )
        case 'NS':
            samples, extras = NSFit(model_args, rng_key)
        case 'MAP':
            print('Warning, Experimental, Do Not Use')
            samples, extras = MAPFit(model_args, rng_key)
        case _:
            raise ValueError(f'Unknown backend: {backend}')

    # Plot the results
    plotResults(config, rows, model_args, samples, output_directory)

    # Save the results
    saveResults(config, rows, model_args, samples, extras, output_directory)


def NIRSpecModelArgs(config: dict, rows: Table, spectra_directory: str) -> Tuple:
    """
    Get the model arguments for the NIRSpec data.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    rows : Table
        Table of the rows

    Returns
    -------
    tuple
        Model arguments
    """

    # Load the spectra
    spectra = NIRSpecSpectra(rows, spectra_directory)

    # Restrict config to what we have coverage of
    config = utils.restrictConfig(config, spectra)

    # If the config is empty, skip
    if len(config['Groups']) == 0:
        raise ValueError('No Line Coverage')

    # Generate Parameter Matrices
    matrices, linetypes_all = parameters.configToMatrices(config)

    # Compute Continuum Regions and Initial Guesses
    cont_regs, cont_guesses = initial.computeContinuumRegions(config, spectra)

    # Compute Line Centers and Equalized estimates
    line_centers, line_estimates_eq = initial.linesFluxesGuess(
        config, spectra, cont_regs, cont_guesses
    )

    # Restrict spectra to continuum regions and rescale errorbars in each region
    spectra.restrictAndRescale(config, cont_regs)

    # Skip if no data
    if len(spectra.spectra) == 0:
        raise ValueError('No Valid Data')

    # Model Args
    return config, (
        spectra,
        matrices,
        linetypes_all,
        line_centers,
        line_estimates_eq,
        cont_regs,
        cont_guesses,
    )


def MCMCFit(
    model_args: tuple,
    rng_key: random.PRNGKey,
    N: int = 500,
    num_warmup: int = 250,
    verbose=True,
) -> Tuple[Dict, Dict]:
    """
    Fit the NIRSpec data with MCMC.

    Parameters
    ----------
    model_args : tuple
        Model Arguements
    rng_key : random.PRNGKey
        JAX random key
    N : int, optional
        Number of samples, by default 500
    verbose : bool, optional
        Verbose, by default True

    Returns
    -------
    infer.MCMC
        MCMC object
    """

    # MCMC
    kernel = infer.NUTS(multiSpecModel)
    mcmc = infer.MCMC(
        kernel, num_samples=N, num_warmup=num_warmup, progress_bar=verbose
    )
    mcmc.run(rng_key, *model_args)

    # Get the samples
    samples = mcmc.get_samples()

    # Compute relevant probabilities
    logL = computeProbs(samples, model_args)

    # Compute the WAIC with numerical stability
    # Use logsumexp trick to avoid underflow: log(mean(exp(x))) = logsumexp(x) - log(n)
    # WAIC = -2 * (log(mean(exp(logL))) - var(logL))
    n_samples = logL.shape[0]
    
    # Center logL to avoid overflow: subtract max along sample axis for each data point
    logL_max = logL.max(axis=0, keepdims=True)
    logL_centered = logL - logL_max
    
    # Compute log(mean(exp(logL))) = log(sum(exp(logL_centered))) - log(n) + max
    # Use logsumexp: log(sum(exp(x))) ≈ max(x) + log(sum(exp(x - max(x))))
    exp_sum = np.exp(logL_centered).sum(axis=0)
    log_mean_likelihood = np.log(exp_sum + 1e-300) - np.log(n_samples) + logL_max.squeeze()
    
    # Compute variance, handling edge cases
    logL_var = logL.var(axis=0, ddof=1)
    # Replace any NaN or inf values with 0
    logL_var = np.nan_to_num(logL_var, nan=0.0, posinf=0.0, neginf=0.0)
    
    waic = -2 * (log_mean_likelihood.sum() - logL_var.sum())
    
    # Handle NaN/inf in final WAIC
    if not np.isfinite(waic):
        waic = np.nan
    
    extras = {'WAIC': waic}

    return samples, extras


def NSFit(
    model_args: tuple, rng_key: random.PRNGKey, N: int = 1000
) -> Tuple[Dict, Dict]:
    """
    Fit the NIRSpec data with Nested Sampling.

    Parameters
    ----------
    model_args : tuple

    Returns
    -------
    NestedSampler
    """

    from numpyro.contrib.nested_sampling import NestedSampler

    # Get number of variables
    with trace() as tr:
        with seed(multiSpecModel, rng_seed=rng_key):
            multiSpecModel(*model_args)
    nv = sum(
        [
            v['value'].size
            for v in tr.values()
            if v['type'] == 'sample' and not v['is_observed']
        ]
    )

    # Nested Sampling
    constructor_kwargs = {'num_live_points': 50 * (nv + 1), 'max_samples': 50000}
    termination_kwargs = {'dlogZ': 0.01}
    NS = NestedSampler(
        model=multiSpecModel,
        constructor_kwargs=constructor_kwargs,
        termination_kwargs=termination_kwargs,
    )
    NS.run(rng_key, *model_args)

    # Get the sample
    samples = NS.get_samples(rng_key, N)

    # Compute relevant probabilities
    _ = computeProbs(samples, model_args)

    # Add log evidence to samples
    extras = {
        'logZ': float(NS._results.log_Z_mean),
        'logZ_err': float(NS._results.log_Z_uncert),
    }

    return samples, extras


def MAPFit(
    model_args: tuple, rng_key: random.PRNGKey, N: int = 1000
) -> Tuple[Dict, Dict]:
    """
    Fit the NIRSpec data with Maximum A Posteriori estimation.

    Parameters
    ----------
    model_args : tuple
        Model arguments
    rng_key : random.PRNGKey
        JAX random key
    num_steps : int, optional
        Number of optimization steps

    Returns
    -------
    Tuple[Dict, Dict]
        Samples and extras dictionaries
    """

    # MAP Estimator
    svi = infer.SVI(
        multiSpecModel,
        infer.autoguide.AutoDelta(multiSpecModel),
        optim.Adam(step_size=1e-2),
        loss=infer.Trace_ELBO(),
    )

    # Run the optimization
    svi_result = svi.run(rng_key, N, *model_args)
    params, losses = svi_result.params, svi_result.losses
    params = {k.removesuffix('_auto_loc'): v for k, v in params.items()}

    # Get trace
    traced_model = trace(substitute(multiSpecModel, data=params)).get_trace(*model_args)

    # Create compatible samples dictionary
    samples = {
        name: jnp.array(site['value'])[None, ...]  # Add sample dimension
        for name, site in traced_model.items()
        if site['type'] in ['deterministic', 'sample']
        and not site.get('is_observed', False)
    }

    return samples, {'losses': losses}


def computeProbs(samples: dict, model_args: tuple) -> np.ndarray:
    # Compute the log likelihood
    logLs = infer.util.log_likelihood(multiSpecModel, samples, *model_args)
    for k, v in logLs.items():
        samples[k] = v
    logL = np.hstack([p for p in logLs.values()])  # Likelihood Matrix
    samples['logL'] = logL.sum(1)

    # Compute the log density
    logP = vmap(lambda s: infer.util.log_density(multiSpecModel, model_args, {}, s)[0])(
        samples
    )
    samples['logP'] = np.array(logP)

    return logL


def saveResults(config, rows, model_args, samples, extras, output_dir) -> None:
    # Get config name
    cname = '_' + config['Name'] if config['Name'] else ''

    # Get common filename
    os.makedirs(f'{output_dir}/Results/', exist_ok=True)
    savename = f'{output_dir}/Results/{rows[0]["root"]}-{rows[0]["srcid"]}{cname}'

    # Unpack model args
    spectra, _, _, _, _, cont_regs, _ = model_args

    # Correct sample units
    samples['flux_all'] = samples['flux_all'] * (spectra.fλ_unit * spectra.λ_unit).to(
        u.Unit(1e-20 * u.erg / (u.cm * u.cm * u.s))
    )
    samples['ew_all'] = samples['ew_all'] * spectra.λ_unit.to(u.AA)

    # Add spectra wavelength to samples
    for spectrum in spectra.spectra:
        samples[f'{spectrum.name}_wavelength'] = spectrum.wave

    # Create outputs
    colnames = [
        n
        for n in ['lsf_scale', 'PRISM_flux', 'PRISM_offset', 'logL', 'logP']
        if n in samples.keys()
    ]
    out = Table([samples[name] for name in colnames], names=colnames)

    # Add continuum regions and error scales to samples
    samples['cont_regs'] = np.array(cont_regs)
    samples.update(
        {
            f'{spectrum.name}_errscales': np.array(spectrum.errscales)
            for spectrum in spectra.spectra
        }
    )

    # Save all samples as npz
    np.savez(f'{savename}_full.npz', **samples)

    # Get names of the lines
    # TODO: Better sanitization of line names?
    line_names = [
        re.sub(
            r'[\[\]]',
            '',
            f'{species["Name"]}_{species["LineType"]}_{line["Wavelength"]}',
        )
        for _, group in config['Groups'].items()
        for species in group['Species']
        for line in species['Lines']
    ]

    # Append line parameter samples
    for colname, unit in zip(
        ['redshift', 'flux', 'fwhm', 'ew'],
        [
            u.dimensionless_unscaled,
            u.Unit(1e-20 * u.erg / u.cm**2 / u.s),
            u.km / u.s,
            u.AA,
        ],
    ):
        data = np.array(samples[f'{colname}_all'].T.tolist()) * unit
        out_part = Table(data.T, names=[f'{line}_{colname}' for line in line_names])
        out = hstack([out, out_part])
    
    # Add total and narrow component fluxes
    # Group lines by species name and wavelength (ignoring line type)
    flux_data = np.array(samples['flux_all'].T.tolist())
    
    # Parse line names to extract (species, wavelength) pairs
    line_groups = {}
    for idx, line_name in enumerate(line_names):
        # Format: "Species_LineType_Wavelength"
        parts = line_name.rsplit('_', 2)
        if len(parts) == 3:
            species, linetype, wavelength = parts
            key = f'{species}_{wavelength}'
            if key not in line_groups:
                line_groups[key] = {'narrow': [], 'broad': [], 'other': []}
            
            # Categorize by line type
            if linetype in ['emission', 'narrow']:
                line_groups[key]['narrow'].append(idx)
            elif linetype == 'broad':
                line_groups[key]['broad'].append(idx)
            else:
                line_groups[key]['other'].append(idx)
    
    # Compute total and narrow fluxes for each unique line
    flux_unit = u.Unit(1e-20 * u.erg / u.cm**2 / u.s)
    total_flux_cols = []
    narrow_flux_cols = []
    
    for key, indices in sorted(line_groups.items()):
        narrow_indices = indices['narrow']
        broad_indices = indices['broad']
        other_indices = indices['other']
        all_indices = narrow_indices + broad_indices + other_indices
        
        # Compute narrow flux (sum of all narrow/emission components)
        if narrow_indices:
            narrow_flux = np.sum(flux_data[narrow_indices, :], axis=0) * flux_unit
            narrow_flux_cols.append((f'{key}_narrow_flux', narrow_flux))
        
        # Compute total flux (narrow + broad + other) if there are multiple components
        if len(all_indices) > 1:
            total_flux = np.sum(flux_data[all_indices, :], axis=0) * flux_unit
            total_flux_cols.append((f'{key}_total_flux', total_flux))
    
    # Add narrow flux columns
    if narrow_flux_cols:
        narrow_names = [col[0] for col in narrow_flux_cols]
        # Extract values and units - each col[1] is a Quantity with shape (n_samples,)
        narrow_values = [col[1].value for col in narrow_flux_cols]
        narrow_unit = narrow_flux_cols[0][1].unit  # Get unit from first column
        # Stack arrays: shape will be (n_columns, n_samples), then transpose to (n_samples, n_columns)
        narrow_data = np.array(narrow_values).T * narrow_unit
        out_part = Table(narrow_data, names=narrow_names)
        out = hstack([out, out_part])
    
    # Add total flux columns
    if total_flux_cols:
        total_names = [col[0] for col in total_flux_cols]
        # Extract values and units - each col[1] is a Quantity with shape (n_samples,)
        total_values = [col[1].value for col in total_flux_cols]
        total_unit = total_flux_cols[0][1].unit  # Get unit from first column
        # Stack arrays: shape will be (n_columns, n_samples), then transpose to (n_samples, n_columns)
        total_data = np.array(total_values).T * total_unit
        out_part = Table(total_data, names=total_names)
        out = hstack([out, out_part])

    # Append LSF samples
    for spectrum in spectra.spectra:
        data = np.array(samples[f'{spectrum.name}_lsf'].T.tolist()) * spectra.λ_unit
        out_part = Table(
            data.T, names=[f'{spectrum.name}_{line}_lsf' for line in line_names]
        )
        out = hstack([out, out_part])

    # Create extra table
    extra = Table([[v] for v in extras.values()], names=extras.keys())

    # Create HDUList
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(out, name='PARAMS'),
            fits.BinTableHDU(extra, name='EXTRAS'),
        ]
    )

    # Save the summary
    hdul.writeto(f'{savename}_summary.fits', overwrite=True)

    # Create Summary CSV
    qs = [0.16, 0.5, 0.84]
    # Compute quantiles for main table
    df_out = out.to_pandas().quantile(qs).T
    df_out.columns = ['P16', 'P50', 'P84']
    
    # For extra table (single values), just repeat the value for all quantiles
    df_extra = pd.DataFrame(index=extras.keys(), columns=['P16', 'P50', 'P84'])
    for key, val in extras.items():
        # Handle NaN/inf values
        if np.isfinite(val):
            df_extra.loc[key] = [val, val, val]
        else:
            df_extra.loc[key] = [np.nan, np.nan, np.nan]
    
    # Concatenate and save
    df = pd.concat([df_out, df_extra], axis=0)
    df.to_csv(f'{savename}_summary.csv')


def plotResults(
    config: list, rows: Table, model_args: tuple, samples: dict, output_dir: str
) -> None:
    """
    Plot the results of the sampling.

    Parameters
    ----------
    savedir : str
        Directory to save the plots
    config: list
        Configuration list
    rows : Table
        Table of the rows
    model_args : tuple
        Arguments for the model
    samples : dict
        Samples from the MCMC


    Returns
    -------
    None

    """
    # Get config name
    cname = '_' + config['Name'] if config['Name'] else ''

    os.makedirs(f'{output_dir}/Plots/', exist_ok=True)

    # Unpack model arguements
    spectra, _, _, line_centers, _, cont_regs, _ = model_args

    # Get the number of spectra and regions
    Nspec, Nregs = len(spectra.spectra), len(cont_regs)

    # Plotting
    figsize = (7.5 * Nregs, 6 * Nspec)
    fig, axes = pyplot.subplots(
        Nspec, Nregs, figsize=figsize, sharex='col', constrained_layout=True
    )
    # fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Ensure axes is always a 2D array
    if Nspec == 1 and Nregs == 1:
        axes = np.array([[axes]])  # Convert single Axes object to a 2D array
    elif Nspec == 1 or Nregs == 1:
        axes = np.atleast_2d(axes).reshape(Nspec, Nregs)  # Convert 1D array to 2D array

    # Build mapping of line indices to line types and names for component plotting
    line_info = []
    for _, group in config['Groups'].items():
        for species in group['Species']:
            for line in species['Lines']:
                line_info.append({
                    'name': species['Name'],
                    'linetype': species['LineType'],
                    'wavelength': line['Wavelength']
                })
    
    # Plot the spectra
    for i, spectrum in enumerate(spectra.spectra):
        # Get the spectrum
        _, wave, _, flux, err = spectrum()
        
        # Get line components and continuum for this spectrum
        lines_data = samples.get(f'{spectrum.name}_lines', None)  # Shape: [n_samples, n_pixels, n_lines]
        continuum_data = samples.get(f'{spectrum.name}_cont', None)  # Shape: [n_samples, n_pixels]
        flux_scale_key = f'{spectrum.name}_flux'
        flux_scale_data = samples.get(flux_scale_key, np.array([1.0]))

        for j, ax in enumerate(axes[i]):
            # Get the continuum region
            cont_reg = cont_regs[j]
            mask = jnp.logical_and(wave > cont_reg[0], wave < cont_reg[1])

            # Plot the spectrum
            ax.plot(wave[mask], flux[mask], color='k', ds='steps-mid', label='Data', zorder=10)

            # Plot errorbars on the spectrum
            ax.errorbar(wave[mask], flux[mask], yerr=err[mask], fmt='none', color='k', alpha=0.3, zorder=9)

            # Get best fit index
            best_idx = samples['logP'].argmax()
            
            # Plot continuum if available
            if continuum_data is not None:
                cont_best = continuum_data[best_idx]
                flux_scale_best = flux_scale_data[best_idx] if len(flux_scale_data) > best_idx else 1.0
                ax.plot(
                    wave[mask],
                    flux_scale_best * cont_best[mask],
                    color='gray',
                    alpha=0.6,
                    linestyle='--',
                    linewidth=1.5,
                    drawstyle='steps-mid',
                    label='Continuum',
                    zorder=5
                )
            
            # Plot individual line components if available
            if lines_data is not None:
                # Group lines by type
                narrow_lines = []
                broad_lines = []
                other_lines = []
                
                for line_idx, info in enumerate(line_info):
                    if line_idx >= lines_data.shape[2]:
                        continue
                    line_component = lines_data[best_idx, :, line_idx]  # Shape: [n_pixels]
                    flux_scale_best = flux_scale_data[best_idx] if len(flux_scale_data) > best_idx else 1.0
                    line_flux = flux_scale_best * line_component[mask]
                    
                    if info['linetype'] == 'emission' or info['linetype'] == 'narrow':
                        narrow_lines.append((line_flux, info))
                    elif info['linetype'] == 'broad':
                        broad_lines.append((line_flux, info))
                    else:
                        other_lines.append((line_flux, info))
                
                # Plot narrow components
                if narrow_lines:
                    narrow_total = np.sum([l[0] for l in narrow_lines], axis=0)
                    ax.plot(
                        wave[mask],
                        narrow_total,
                        color='blue',
                        alpha=0.7,
                        linewidth=1.5,
                        drawstyle='steps-mid',
                        label='Narrow components',
                        zorder=6
                    )
                
                # Plot broad components
                if broad_lines:
                    broad_total = np.sum([l[0] for l in broad_lines], axis=0)
                    ax.plot(
                        wave[mask],
                        broad_total,
                        color='green',
                        alpha=0.7,
                        linewidth=1.5,
                        drawstyle='steps-mid',
                        label='Broad components',
                        zorder=7
                    )

            # Plot the models (all samples)
            model = samples[f'{spectrum.name}_model']
            for k in range(model.shape[0]):
                ax.plot(
                    wave[mask],
                    model[k][mask],
                    color='#E20134',
                    alpha=np.clip(5 / len(model), 0.01, 1),
                    ds='steps-mid',
                    zorder=3
                )

            # Plot the best logP model (total)
            m = model[best_idx]
            ax.plot(
                wave[mask],
                m[mask],
                color='#A40122',
                alpha=1,
                lw=2,
                ds='steps-mid',
                label='Total model',
                zorder=8
            )
            
            # Add legend for first subplot
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

            # Label the axes
            if j == 0:
                ax.set(ylabel=f'{spectrum.name}')
            ax.set(xlim=cont_reg)

            # Add rest frame axis
            rest_ax = ax.secondary_xaxis(
                'top',
                functions=(
                    lambda x: x / (1 + spectra.redshift_initial),
                    lambda x: x * (1 + spectra.redshift_initial),
                ),
            )

            # Turn off top xticklabels in the middle
            if i > 0:
                rest_ax.set(xticklabels=[])

            # Turn off top xticks
            ax.tick_params(axis='x', which='both', top=False)

            # Line Labels
            for line in jnp.unique(line_centers):
                line = line * (1 + spectra.redshift_initial)
                if line < cont_reg[0] or line > cont_reg[1]:
                    continue
                ax.axvline(line, color='k', linestyle='--', alpha=0.5)

    # Set superlabels
    fig.supylabel(
        rf'$f_\lambda$ [{spectrum.fλ_unit.to_string(format="latex", fraction=False)}]'
    )
    fig.supxlabel(
        rf'$\lambda$ (Observed) [{spectrum.λ_unit.to_string(format="latex", fraction=False)}]',
        y=-0.01,
        va='center',
        fontsize='medium',
    )
    fig.suptitle(
        rf'$\lambda$ (Rest) [{spectrum.λ_unit:latex_inline}]',
        y=1.015,
        va='center',
        fontsize='medium',
    )
    fig.text(
        0.5,
        1.05,
        f'{rows[0]["srcid"]} ({rows[0]["root"]}): $z = {spectrum.redshift_initial:.3f}$',
        ha='center',
        va='center',
        fontsize='large',
    )

    # Show the plot
    fig.savefig(
        os.path.join(
            f'{output_dir}/Plots',
            f'{rows[0]["root"]}-{rows[0]["srcid"]}{cname}_fit.png',
        ),
        dpi=300,
    )
    pyplot.close(fig)
