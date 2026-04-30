"""Sampling implementation using emcee

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bayesian import emulation

logger = logging.getLogger(__name__)


def _run_using_emcee(
    config: MCMCConfig,
    emulation_config: emulation.EmulationConfig,
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

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        initializer=log_posterior.initialize_pool_variables,
        initargs=[
            parameter_min,
            parameter_max,
            emulation_config,
            emulation_results,
            experimental_results,
            emulator_cov_unexplained,
        ],
    ) as pool:
        # Construct sampler (we create a dummy daughter class from emcee.EnsembleSampler, to add some logging info)
        # Note: we pass the emulators and experimental data as args to the log_posterior function
        logger.info("Initializing sampler...")
        sampler = LoggingEnsembleSampler(
            config.n_walkers,
            parameter_ndim,
            log_posterior.log_posterior,
            # args=[min, max, emulation_config, emulation_results, experimental_results, emulator_cov_unexplained],
            kwargs={"set_to_infinite_outside_bounds": True},
            pool=pool,
        )

        # Generate random starting positions for each walker
        rng = np.random.default_rng()
        random_pos = rng.uniform(parameter_min, parameter_max, (config.n_walkers, parameter_ndim))

        # Run first half of burn-in
        # NOTE-STAT: This code doesn't support not doing burn in
        logger.info(f"Parallelizing over {pool._processes} processes...")  # type: ignore[attr-defined]
        logger.info("Starting initial burn-in...")
        nburn0 = config.n_burn_steps // 2
        sampler.run_mcmc(random_pos, nburn0, n_logging_steps=config.n_logging_steps)

        # Reposition walkers to the most likely points in the chain, then run the second half of burn-in.
        # This significantly accelerates burn-in and helps prevent stuck walkers.
        logger.info("Resampling walker positions...")
        X0 = sampler.flatchain[np.unique(sampler.flatlnprobability, return_index=True)[1][-config.n_walkers :]]
        sampler.reset()
        X0 = sampler.run_mcmc(X0, config.n_burn_steps - nburn0, n_logging_steps=config.n_logging_steps)[0]
        sampler.reset()
        logger.info("Burn-in complete.")

        # Production samples
        logger.info("Starting production...")
        sampler.run_mcmc(X0, config.n_sampling_steps, n_logging_steps=config.n_logging_steps)

        # Write to file
        logger.info("Writing chain to file...")
        output_dict = {}
        output_dict["chain"] = sampler.get_chain()
        output_dict["acceptance_fraction"] = sampler.acceptance_fraction
        output_dict["log_prob"] = sampler.get_log_prob()
        try:
            output_dict["autocorrelation_time"] = sampler.get_autocorr_time()
        except Exception as e:
            output_dict["autocorrelation_time"] = None
            logger.info(f"Could not compute autocorrelation time: {e!s}")
        # If closure test, save the design point parameters and experimental pseudodata
        if closure_index >= 0:
            design_point = data_IO.design_array_from_h5(
                config.output_dir, filename="observables.h5", validation_set=True
            )[closure_index]
            output_dict["design_point"] = design_point
            # Dec 2025: Replace the experimental_pseudodata with the cleaned_results below.
            # TODO(RJE): Confirm this works as expected.
            # output_dict['experimental_pseudodata'] = experimental_results

            cleaned_results = {}

            # Copy essential arrays with proper dtypes
            for key in ["y", "y_err_stat"]:
                if key in experimental_results:
                    cleaned_results[key] = np.array(experimental_results[key], dtype=np.float64)

            # Handle systematic uncertainties
            if "y_err_syst" in experimental_results:
                cleaned_results["y_err_syst"] = np.array(experimental_results["y_err_syst"], dtype=np.float64)

            # Handle systematic names as clean strings
            if "systematic_names" in experimental_results:
                cleaned_results["systematic_names"] = [str(name) for name in experimental_results["systematic_names"]]

            # Copy other simple fields
            for key in ["y_err"]:  # Include any other simple fields you need
                if key in experimental_results and key not in cleaned_results:
                    cleaned_results[key] = experimental_results[key]

            # Replace with cleaned version
            experimental_results = cleaned_results

        data_IO.write_dict_to_h5(output_dict, config.mcmc_output_dir, "mcmc.h5", verbose=True)

        # Save the sampler to file as well, in case we want to access it later
        #   e.g. sampler.get_chain(discard=n_burn_steps, thin=thin, flat=True)
        # Note that currently we use sampler.reset() to discard the burn-in and reposition
        #   the walkers (and free memory), but it prevents us from plotting the burn-in samples.
        with Path(config.sampler_outputfile).open("wb") as f:
            pickle.dump(sampler, f)

        logger.info("Done.")


####################################################################################################################
class LoggingEnsembleSampler(emcee.EnsembleSampler):
    """
    Add some logging to the emcee.EnsembleSampler class.
    Inherit from: https://emcee.readthedocs.io/en/stable/user/sampler/
    """

    # ---------------------------------------------------------------
    def run_mcmc(self, X0, n_sampling_steps, n_logging_steps=100, **kwargs):
        """
        Run MCMC with logging every 'logging_steps' steps (default: log every 100 steps).
        """
        logger.info(f"  running {self.nwalkers} walkers for {n_sampling_steps} steps")
        for n, result in enumerate(self.sample(X0, iterations=n_sampling_steps, **kwargs), start=1):
            if n % n_logging_steps == 0 or n == n_sampling_steps:
                af = self.acceptance_fraction
                logger.info(
                    f"  step {n}: acceptance fraction: mean {af.mean()}, std {af.std()}, min {af.min()}, max {af.max()}"
                )

        return result
