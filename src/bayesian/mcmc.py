'''
Module related to MCMC, with functionality to compute posterior for a given analysis run

The main functionalities are:
 - run_mcmc() performs MCMC and returns posterior
 - credible_interval() compute credible interval for a given posterior

A configuration class MCMCConfig provides simple access to emulation settings

authors: J.Mulligan, R.Ehlers
Based in part on JETSCAPE/STAT code.
'''
from __future__ import annotations

import logging
import multiprocessing
import pickle
from pathlib import Path

import os

import emcee
import numpy as np
import numpy.typing as npt
import yaml

from bayesian import common_base, data_IO, log_posterior
from bayesian.emulation import base

logger = logging.getLogger(__name__)


####################################################################################################################
def run_mcmc(config: MCMCConfig, closure_index: int =-1) -> None:
    '''
    Run MCMC to compute posterior

    :param MCMCConfig config: Instance of MCMCConfig
    :param int closure_index: Index of validation design point to use for MCMC closure. Off by default.
                              If non-negative index is specified, will construct pseudodata from the design point
                              and use that for the closure test.
    '''

    # Get parameter names and min/max
    names = config.analysis_config['parameterization'][config.parameterization]['names']
    parameter_min = config.analysis_config['parameterization'][config.parameterization]['min']
    parameter_max = config.analysis_config['parameterization'][config.parameterization]['max']
    ndim = len(names)

    # Load emulators
    emulation_config = base.EmulatorOrganizationConfig.from_config_file(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        analysis_config=config.analysis_config,
        config_file=config.config_file,
    )
    emulation_results = emulation_config.read_all_emulator_groups()

    # Pre-compute the predictive variance due to PC truncation, since it is independent of theta.
    emulator_cov_unexplained = base.compute_emulator_cov_unexplained(emulation_config, emulation_results)

    # Load experimental data into arrays: experimental_results['y'/'y_err'] (n_features,)
    # In the case of a closure test, we use the pseudodata from the validation design point
    experimental_results = data_IO.data_array_from_h5(config.output_dir, 'observables.h5', pseudodata_index=closure_index, observable_filter=emulation_config.observable_filter)

    if config.mcmc_package == "emcee":
        _run_using_emcee(
            config,
            emulation_config,
            emulation_results,
            emulator_cov_unexplained,
            experimental_results,
            parameter_min,
            parameter_max,
            ndim,
            closure_index=closure_index,
        )
    elif config.mcmc_package == "pocoMC":
        _run_using_pocoMC(
            config,
            emulation_config,
            emulation_results,
            emulator_cov_unexplained,
            experimental_results,
            parameter_min,
            parameter_max,
            ndim,
            closure_index=closure_index,
        )
    else:
        msg = f"Invalid MCMC sampler: {config.mcmc_package}"
        raise ValueError(msg)



####################################################################################################################
def credible_interval(samples, confidence=0.9, interval_type='quantile'):
    '''
    Compute the credible interval for an array of samples.

    TODO: one could also call the versions in pymc3 or arviz

    :param 1darray samples: Array of samples
    :param float confidence: Confidence level (default 0.9)
    :param str type: Type of credible interval to compute. Options are:
                        'hpd' - highest-posterior density
                        'quantile' - quantile interval
    '''

    if interval_type == 'hpd':
        # number of intervals to compute
        nci = int((1 - confidence)*samples.size)
        # find highest posterior density (HPD) credible interval i.e. the one with minimum width
        argp = np.argpartition(samples, [nci, samples.size - nci])
        cil = np.sort(samples[argp[:nci]])   # interval lows
        cih = np.sort(samples[argp[-nci:]])  # interval highs
        ihpd = np.argmin(cih - cil)
        ci = cil[ihpd], cih[ihpd]

    elif interval_type == 'quantile':
        cred_range = [(1-confidence)/2, 1-(1-confidence)/2]
        ci = np.quantile(samples, cred_range)

    return ci

####################################################################################################################
def map_parameters(posterior, method='quantile'):
    '''
    Compute the MAP parameters

    :param 1darray posterior: Array of samples
    :param str method: Method used to compute MAP. Options are:
                        'quantile' - take a narrow quantile interval and compute mean of parameters in that interval
    :return 1darray map_parameters: Array of MAP parameters
    '''

    if method == 'quantile':
        central_quantile = 0.01
        lower_bounds = np.quantile(posterior, 0.5-central_quantile/2, axis=0)
        upper_bounds = np.quantile(posterior, 0.5+central_quantile/2, axis=0)
        mask = (posterior >= lower_bounds) & (posterior <= upper_bounds)
        map_parameters = np.array([posterior[mask[:,i],i].mean() for i in range(posterior.shape[1])])

    return map_parameters


def _run_using_emcee(
    config: MCMCConfig,
    emulation_config: base.EmulatorOrganizationConfig,
    emulation_results: dict[str, dict[str, npt.NDArray[np.float64]]],
    emulator_cov_unexplained: dict,
    experimental_results: dict,
    parameter_min: npt.NDArray[np.float64],
    parameter_max: npt.NDArray[np.float64],
    parameter_ndim: int,
    closure_index: int,
) -> None:
    """Run emcee-based MCMC.

    Markov chain Monte Carlo model calibration using the `affine-invariant ensemble
    sampler (emcee) <http://dfm.io/emcee>`.

    This is separated out so we can use potentially select other MCMC packages.

    Args:
        config: MCMC config
        emulation_config: Emulation configuration
        emulation_results: Results from the emulator.
        emulator_cov_unexplained: Covariance of the emulator unexplained variance.
        experimental_results: Experimental results.
        parameter_min: Minimum parameter values.
        parameter_max: Maximum parameter values.
        parameter_ndim: Number of dimensions of the parameters.
        closure_index: Index of the closure test design point. If negative, no closure test is performed.
    """
    # TODO: By default the chain will be stored in memory as a numpy array
    #       If needed we can create a h5py dataset for compression/chunking

    # We can use multiprocessing in emcee to parallelize the independent walkers
    # NOTE: We need to use `spawn` rather than `fork` on linux. Otherwise, the some of the caching mechanisms
    #       (eg. used in learning the emulator group mapping doesn't work)
    # NOTE: We use `get_context` here to avoid having to globally specify the context. Plus, it then should be fine
    #       to repeated call this function. (`set_context` can only be called once - otherwise, it's a runtime error).
    # Persist the static covariance pieces once on the master before any worker process
    # is created, so workers don't race on covariance_matrices.pkl.
    log_posterior.save_covariance_matrices_for_plotting(experimental_results, emulation_config.output_dir)

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(
        initializer=log_posterior.initialize_pool_variables,
        initargs=[
            parameter_min, parameter_max, emulation_config, emulation_results, experimental_results, emulator_cov_unexplained
        ]) as pool:

        # Construct sampler (we create a dummy daughter class from emcee.EnsembleSampler, to add some logging info)
        # Note: we pass the emulators and experimental data as args to the log_posterior function
        logger.info('Initializing sampler...')
        sampler = LoggingEnsembleSampler(config.n_walkers, parameter_ndim, log_posterior.log_posterior,
                                         #args=[min, max, emulation_config, emulation_results, experimental_results, emulator_cov_unexplained],
                                         kwargs={'set_to_infinite_outside_bounds': True},
                                         pool=pool)

        # Generate random starting positions for each walker
        rng = np.random.default_rng()
        random_pos = rng.uniform(parameter_min, parameter_max, (config.n_walkers, parameter_ndim))

        # Run first half of burn-in
        # NOTE-STAT: This code doesn't support not doing burn in
        logger.info(f'Parallelizing over {pool._processes} processes...')  # type: ignore[attr-defined]
        logger.info('Starting initial burn-in...')
        nburn0 = config.n_burn_steps // 2
        sampler.run_mcmc(random_pos, nburn0, n_logging_steps=config.n_logging_steps)

        # Reposition walkers to the most likely points in the chain, then run the second half of burn-in.
        # This significantly accelerates burn-in and helps prevent stuck walkers.
        logger.info('Resampling walker positions...')
        X0 = sampler.flatchain[np.unique(sampler.flatlnprobability, return_index=True)[1][-config.n_walkers:]]
        sampler.reset()
        X0 = sampler.run_mcmc(X0, config.n_burn_steps - nburn0, n_logging_steps=config.n_logging_steps)[0]
        sampler.reset()
        logger.info('Burn-in complete.')

        # Production samples
        logger.info('Starting production...')
        sampler.run_mcmc(X0, config.n_sampling_steps, n_logging_steps=config.n_logging_steps)

        # Write to file
        logger.info('Writing chain to file...')
        output_dict = {}
        output_dict['chain'] = sampler.get_chain()
        output_dict['acceptance_fraction'] = sampler.acceptance_fraction
        output_dict['log_prob'] = sampler.get_log_prob()
        try:
            output_dict['autocorrelation_time'] = sampler.get_autocorr_time()
        except Exception as e:
            output_dict['autocorrelation_time'] = None
            logger.info(f"Could not compute autocorrelation time: {e!s}")
        # If closure test, save the design point parameters and experimental pseudodata
        if closure_index >= 0:
            design_point = data_IO.design_array_from_h5(config.output_dir, filename='observables.h5', validation_set=True)[closure_index]
            output_dict['design_point'] = design_point
    
            cleaned_results = {}
            
            # Copy essential arrays with proper dtypes
            for key in ['y', 'y_err_stat']:
                if key in experimental_results:
                    cleaned_results[key] = np.array(experimental_results[key], dtype=np.float64)
            
            # Handle systematic uncertainties
            if 'y_err_syst' in experimental_results:
                cleaned_results['y_err_syst'] = np.array(experimental_results['y_err_syst'], dtype=np.float64)
            
            # Handle systematic names as clean strings
            if 'systematic_names' in experimental_results:
                cleaned_results['systematic_names'] = [str(name) for name in experimental_results['systematic_names']]
            
            # Copy other simple fields
            for key in ['y_err']:  # Include any other simple fields you need
                if key in experimental_results and key not in cleaned_results:
                    cleaned_results[key] = experimental_results[key]
            
            # Replace with cleaned version
            experimental_results = cleaned_results
            
        data_IO.write_dict_to_h5(output_dict, config.mcmc_output_dir, 'mcmc.h5', verbose=True)

        # Save the sampler to file as well, in case we want to access it later
        #   e.g. sampler.get_chain(discard=n_burn_steps, thin=thin, flat=True)
        # Note that currently we use sampler.reset() to discard the burn-in and reposition
        #   the walkers (and free memory), but it prevents us from plotting the burn-in samples.
        with Path(config.sampler_outputfile).open('wb') as f:
            pickle.dump(sampler, f)

        logger.info('Done.')

####################################################################################################################
class LoggingEnsembleSampler(emcee.EnsembleSampler):
    '''
    Add some logging to the emcee.EnsembleSampler class.
    Inherit from: https://emcee.readthedocs.io/en/stable/user/sampler/
    '''

    #---------------------------------------------------------------
    def run_mcmc(self, X0, n_sampling_steps, n_logging_steps=100, **kwargs):
        """
        Run MCMC with logging every 'logging_steps' steps (default: log every 100 steps).
        """
        #logger.info(f'  running {self.nwalkers} walkers for {n_sampling_steps} steps')
        for n, result in enumerate(self.sample(X0, iterations=n_sampling_steps, **kwargs), start=1):
            if n % n_logging_steps == 0 or n == n_sampling_steps:
                af = self.acceptance_fraction
                #logger.info(f'  step {n}: acceptance fraction: mean {af.mean()}, std {af.std()}, min {af.min()}, max {af.max()}')

        return result


def _run_using_pocoMC(
    config: MCMCConfig,
    emulation_config: base.EmulatorOrganizationConfig,
    emulation_results: dict[str, dict[str, npt.NDArray[np.float64]]],
    emulator_cov_unexplained: dict,
    experimental_results: dict,
    parameter_min: npt.NDArray[np.float64],
    parameter_max: npt.NDArray[np.float64],
    parameter_ndim: int,
    closure_index: int,
    n_max_steps: int = -1,
) -> None:
    """ Run with pocoMC.

    This function is based on PocoMC package (version 1.2.1).
    pocoMC is a Preconditioned Monte Carlo (PMC) sampler that uses
    normalizing flows to precondition the target distribution.

    It draws heavily on the wrapper by Hendrick Roch, available at:
    https://github.com/Hendrik1704/GPBayesTools-HIC/blob/0e41660fafaf1ea2beec3a141a9baa466f31e7c2/src/mcmc.py#L939
    """
    # Setup
    import pocomc as pmc
    import scipy.stats

    # Validation
    if n_max_steps < 0:
        # n_max_steps (int): Maximum number of MCMC steps (default is max_steps=10*n_dim).
        n_max_steps = 10 * parameter_ndim

    # Additional possible function parameters, but for now, we don't need to pass it in.
    # random_state (int or None): Initial random seed.
    random_state = None
    # pool (int): Number of processes to use for parallelization (default is ``pool=None``).
    #     If ``pool`` is an integer greater than 1, a ``multiprocessing`` pool is created with the specified number of processes.
    #pool = None

    # pocoMC config
    pocoMC_config = PocoMCConfig(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        analysis_config=config.analysis_config,
        config_file=config.config_file,
    )

    # Setup the prior distributions
    logging.info('Generate the prior class for pocoMC ...')
    prior_distributions = []
    for p_min, p_max in zip(parameter_min, parameter_max, strict=True):
        # NOTE: Assuming uniform prior
        # TODO: Need to update this for c1, c2, and c3, which is uniform in log space.
        prior_distributions.append(scipy.stats.uniform(p_min, p_max))
    prior = pmc.Prior(prior_distributions)

    # Create and run the pocoMC sampler
    # We can use multiprocessing in pocoMC to parallelize the calls to the particles
    # NOTE: We need to use `spawn` rather than `fork` on linux. Otherwise, the some of the caching mechanisms
    #       (eg. used in learning the emulator group mapping doesn't work)
    # NOTE: We use `get_context` here to avoid having to globally specify the context. Plus, it then should be fine
    #       to repeated call this function. (`set_context` can only be called once - otherwise, it's a runtime error).
    # NOTE: I create the pool here rather than using the built-in one because I need to initialize the log_posterior!
    # Persist the static covariance pieces once on the master before any worker process
    # is created, so workers don't race on covariance_matrices.pkl.
    log_posterior.save_covariance_matrices_for_plotting(experimental_results, emulation_config.output_dir)

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(
        initializer=log_posterior.initialize_pool_variables,
        initargs=[
            parameter_min, parameter_max, emulation_config, emulation_results, experimental_results, emulator_cov_unexplained
        ]) as pool:
        logging.info('Starting pocoMC ...')
        sampler = pmc.Sampler(
            prior=prior,
            #likelihood=self.log_likelihood,
            # TODO: Need initialization function...
            likelihood=log_posterior.log_posterior,
            likelihood_kwargs={"set_to_infinite_outside_bounds": False},
            n_effective=pocoMC_config.n_effective,
            n_active=pocoMC_config.n_active,
            n_prior=pocoMC_config.draw_n_prior_samples,
            sample=pocoMC_config.sampler_type,
            n_max_steps=n_max_steps,
            random_state=random_state,
            vectorize=True,
            pool=pool
        )
        sampler.run(n_total=pocoMC_config.n_total_samples, n_evidence=pocoMC_config.n_importance_samples_for_evidence)

    logging.info('Generate the posterior samples ...')
    samples, weights, logl, logp = sampler.posterior() # Weighted posterior samples

    logging.info('Generate the evidence ...')
    logz, logz_err = sampler.evidence() # Bayesian model evidence estimate and uncertainty
    logger.info(f"Log evidence: {logz}")
    logger.info(f"Log evidence error: {logz_err}")

    logging.info('Writing pocoMC chains to file...')
    chain_data = {'chain': samples, 'weights': weights, 'logl': logl,
                    'logp': logp, 'logz': logz, 'logz_err': logz_err}
    with config.mcmc_outputfile.open('wb') as file:
        pickle.dump(chain_data, file)


class PocoMCConfig(common_base.CommonBase):
    """ Configuration class for pocoMC MCMC sampler. """
    def __init__(self, analysis_name="", parameterization="", analysis_config="", config_file="",
                       closure_index=-1, **kwargs):

        self.analysis_name = analysis_name
        self.parameterization = parameterization
        self.analysis_config = analysis_config
        self.config_file = Path(config_file)

        with self.config_file.open() as stream:
            config = yaml.safe_load(stream)

        self.observable_table_dir = config['observable_table_dir']
        self.observable_config_dir = config['observable_config_dir']
        self.observables_filename = config["observables_filename"]

        """

        """
        # NOTE: Do not retrieve this conditionally - if we're asking for it, it's needed.
        try:
            mcmc_configuration = analysis_config["parameters"]["mcmc"]["pocoMC"]
        except KeyError as e:
            msg = "Please provide pocoMC configuration in the analysis configuration."
            raise KeyError(msg) from e

        # n_effective (int): The effective sample size maintained during the run (default is n_ess=1000).
        #self.n_effective = mcmc_configuration.get("n_effective", 1000)
        # 512 is the default from pocoMC
        self.n_effective = mcmc_configuration.get("n_effective", 512)
        # n_active (int): The number of active particles (default is n_active=250). It must be smaller than n_ess.
        self.n_active = mcmc_configuration.get("n_active", 250)
        # Validation
        if self.n_active >= self.n_effective:
            msg = f"n_active ({self.n_active}) must be smaller than n_effective ({self.n_effective})."
            raise ValueError(msg)

        # n_prior (int): Number of prior samples to draw (default is n_prior=2*(n_effective//n_active)*n_active).
        self.draw_n_prior_samples = mcmc_configuration.get("draw_n_prior_samples", 2*(self.n_effective//self.n_active)*self.n_active)
        # sample (str): Type of MCMC sampler to use (default is sample="pcn").
        #     Options are ``"pcn"`` (t-preconditioned Crank-Nicolson) or ``"rwm"`` (Random-walk Metropolis).
        #     t-preconditioned Crank-Nicolson is the default and recommended sampler for PMC as it is more efficient and scales better with the number of parameters.
        self.sampler_type = mcmc_configuration.get("sampler_type", "tpcn")

        # n_total (int): The total number of effectively independent samples to be collected (default is n_total=5000).
        # n_evidence (int): The number of importance samples used to estimate the evidence (default is n_evidence=5000).
        #                     If n_evidence=0, the evidence is not estimated using importance sampling and the SMC estimate is used instead.
        #                     If preconditioned=False, the evidence is estimated using SMC and n_evidence is ignored.
        self.n_total_samples = mcmc_configuration.get("n_total_samples", 5000)
        self.n_importance_samples_for_evidence = mcmc_configuration.get("n_importance_samples_for_evidence", 5000)

        self.output_dir = Path(config['output_dir']) / f'{analysis_name}_{parameterization}'
        self.emulation_outputfile = Path(self.output_dir) / 'emulation.pkl'
        self.mcmc_outputfilename = 'mcmc.h5'
        if closure_index < 0:
            self.mcmc_output_dir = Path(self.output_dir)
        else:
            self.mcmc_output_dir = Path(self.output_dir) / f'closure/results/{closure_index}'
        self.mcmc_outputfile = Path(self.mcmc_output_dir) / 'mcmc.h5'
        self.sampler_outputfile = Path(self.mcmc_output_dir) / 'mcmc_sampler.pkl'

        # Update formatting of parameter names for plotting
        unformatted_names = self.analysis_config['parameterization'][self.parameterization]['names']
        self.analysis_config['parameterization'][self.parameterization]['names'] = [rf'{s}' for s in unformatted_names]


class MCMCConfig(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, analysis_name='', parameterization='', analysis_config='', config_file='',
                       closure_index=-1, **kwargs):

        self.analysis_name = analysis_name
        self.parameterization = parameterization
        self.analysis_config = analysis_config
        self.config_file = Path(config_file)

        with self.config_file.open() as stream:
            config = yaml.safe_load(stream)

        self.observable_table_dir = config['observable_table_dir']
        self.observable_config_dir = config['observable_config_dir']
        self.observables_filename = config["observables_filename"]

        mcmc_configuration = analysis_config["parameters"]["mcmc"]
        # General arguments
        self.mcmc_package = mcmc_configuration.get("mcmc_package", "emcee")
        # emcee specific
        self.n_walkers = mcmc_configuration['n_walkers']
        self.n_burn_steps = mcmc_configuration['n_burn_steps']
        self.n_sampling_steps = mcmc_configuration['n_sampling_steps']
        self.n_logging_steps = mcmc_configuration['n_logging_steps']

        self.output_dir = Path(config['output_dir']) / f'{analysis_name}_{parameterization}'
        self.emulation_outputfile = Path(self.output_dir) / 'emulation.pkl'
        self.mcmc_outputfilename = 'mcmc.h5'
        if closure_index < 0:
            self.mcmc_output_dir = Path(self.output_dir)
        else:
            self.mcmc_output_dir = Path(self.output_dir) / f'closure/results/{closure_index}'
        self.mcmc_outputfile = Path(self.mcmc_output_dir) / 'mcmc.h5'
        self.sampler_outputfile = Path(self.mcmc_output_dir) / 'mcmc_sampler.pkl'

        # Update formatting of parameter names for plotting
        unformatted_names = self.analysis_config['parameterization'][self.parameterization]['names']
        self.analysis_config['parameterization'][self.parameterization]['names'] = [rf'{s}' for s in unformatted_names]
