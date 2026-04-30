"""Sampling implementation using emcee

Canonical implementation of the mc_sampling plugin interface using the affine-invariant
ensemble sampler (emcee).

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
from pathlib import Path
from typing import Any, ClassVar

import attrs
import emcee
import numpy as np
import numpy.typing as npt

from bayesian import data_IO, emulation, log_posterior
from bayesian.mc_sampling import base as mc_sampling_base

logger = logging.getLogger(__name__)

_register_name = "emcee"


@attrs.define
class SamplerSettings:
    """Settings for the emcee affine-invariant ensemble sampler.

    Attributes:
        sampler_name: Name matching _register_name.
        base_settings: Base sampler settings (holds raw config dict).
        n_walkers: Number of ensemble walkers.
        n_burn_steps: Total burn-in steps (split into two stages internally).
        n_sampling_steps: Number of production sampling steps.
        n_logging_steps: Log acceptance fraction every this many steps.
        settings: Raw MCMC config dict.
    """

    sampler_name: ClassVar[str] = "emcee"
    base_settings: mc_sampling_base.BaseSamplerSettings = attrs.field()
    n_walkers: int = attrs.field()
    n_burn_steps: int = attrs.field()
    n_sampling_steps: int = attrs.field()
    n_logging_steps: int = attrs.field()
    settings: dict[str, Any] = attrs.field()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SamplerSettings:
        return cls(
            base_settings=mc_sampling_base.BaseSamplerSettings.from_config(config),
            n_walkers=config["n_walkers"],
            n_burn_steps=config["n_burn_steps"],
            n_sampling_steps=config["n_sampling_steps"],
            n_logging_steps=config["n_logging_steps"],
            settings=config,
        )


def run_sampling(
    config: mc_sampling_base.MCConfig,
    emulation_config: emulation.EmulationConfig,
    emulation_results: dict[str, Any],
    experimental_results: dict[str, Any],
    parameter_min: npt.NDArray[np.float64],
    parameter_max: npt.NDArray[np.float64],
    parameter_ndim: int,
) -> None:
    """Run emcee-based MCMC.

    Markov chain Monte Carlo model calibration using the affine-invariant ensemble
    sampler (emcee). Uses a two-stage burn-in: walkers are reseeded at the most
    probable positions after the first half of burn-in to accelerate convergence.

    Args:
        config: MC sampling configuration (includes output paths, closure_index).
        emulation_config: Emulation configuration.
        emulation_results: Trained emulator results, keyed by emulator group name.
        experimental_results: Experimental data arrays.
        parameter_min: Lower bounds for each parameter.
        parameter_max: Upper bounds for each parameter.
        parameter_ndim: Number of parameters.
    """
    sampler_settings: SamplerSettings = config.sampler_settings

    # NOTE: We need to use `spawn` rather than `fork` on linux. Otherwise, some caching
    #       mechanisms (e.g. used in learning the emulator group mapping) don't work.
    # NOTE: We use `get_context` here to avoid having to globally specify the context.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        initializer=log_posterior.initialize_pool_variables,
        initargs=[
            parameter_min,
            parameter_max,
            emulation_config,
            emulation_results,
            experimental_results,
            {},  # emulator_cov_unexplained: computed dynamically by predict() if needed
        ],
    ) as pool:
        logger.info("Initializing sampler...")
        sampler = LoggingEnsembleSampler(
            sampler_settings.n_walkers,
            parameter_ndim,
            log_posterior.log_posterior,
            kwargs={"set_to_infinite_outside_bounds": True},
            pool=pool,
        )

        rng = np.random.default_rng()
        random_pos = rng.uniform(parameter_min, parameter_max, (sampler_settings.n_walkers, parameter_ndim))

        # First half of burn-in from random positions
        logger.info(f"Parallelizing over {pool._processes} processes...")  # type: ignore[attr-defined]
        logger.info("Starting initial burn-in...")
        nburn0 = sampler_settings.n_burn_steps // 2
        sampler.run_mcmc(random_pos, nburn0, n_logging_steps=sampler_settings.n_logging_steps)

        # Reseed walkers at the most probable positions, then complete burn-in.
        # This significantly accelerates convergence and helps prevent stuck walkers.
        logger.info("Resampling walker positions...")
        X0 = sampler.flatchain[
            np.unique(sampler.flatlnprobability, return_index=True)[1][-sampler_settings.n_walkers :]
        ]
        sampler.reset()
        X0 = sampler.run_mcmc(
            X0, sampler_settings.n_burn_steps - nburn0, n_logging_steps=sampler_settings.n_logging_steps
        )[0]
        sampler.reset()
        logger.info("Burn-in complete.")

        # Production sampling
        logger.info("Starting production...")
        sampler.run_mcmc(X0, sampler_settings.n_sampling_steps, n_logging_steps=sampler_settings.n_logging_steps)

        # Collect results
        logger.info("Writing chain to file...")
        output_dict: dict[str, Any] = {
            "chain": sampler.get_chain(),
            "acceptance_fraction": sampler.acceptance_fraction,
            "log_prob": sampler.get_log_prob(),
        }
        try:
            output_dict["autocorrelation_time"] = sampler.get_autocorr_time()
        except Exception as e:
            output_dict["autocorrelation_time"] = None
            logger.info(f"Could not compute autocorrelation time: {e!s}")

        # For closure tests, save the design point parameters and pseudodata
        if config.closure_index >= 0:
            design_point = data_IO.design_array_from_h5(
                config.output_dir, filename="observables.h5", validation_set=True
            )[config.closure_index]
            output_dict["design_point"] = design_point

            cleaned_results: dict[str, Any] = {}
            for key in ["y", "y_err_stat"]:
                if key in experimental_results:
                    cleaned_results[key] = np.array(experimental_results[key], dtype=np.float64)
            if "y_err_syst" in experimental_results:
                cleaned_results["y_err_syst"] = np.array(experimental_results["y_err_syst"], dtype=np.float64)
            if "systematic_names" in experimental_results:
                cleaned_results["systematic_names"] = [str(name) for name in experimental_results["systematic_names"]]
            for key in ["y_err"]:
                if key in experimental_results and key not in cleaned_results:
                    cleaned_results[key] = experimental_results[key]
            experimental_results = cleaned_results

        config.mcmc_output_dir.mkdir(exist_ok=True, parents=True)
        data_IO.write_dict_to_h5(output_dict, config.mcmc_output_dir, "mcmc.h5", verbose=True)

        with Path(config.sampler_outputfile).open("wb") as f:
            pickle.dump(sampler, f)

        logger.info("Done.")


class LoggingEnsembleSampler(emcee.EnsembleSampler):
    """emcee.EnsembleSampler with periodic acceptance-fraction logging."""

    def run_mcmc(
        self, X0: npt.NDArray[np.float64], n_sampling_steps: int, n_logging_steps: int = 100, **kwargs: Any
    ) -> Any:
        logger.info(f"  running {self.nwalkers} walkers for {n_sampling_steps} steps")
        result = None
        for n, result in enumerate(self.sample(X0, iterations=n_sampling_steps, **kwargs), start=1):
            if n % n_logging_steps == 0 or n == n_sampling_steps:
                af = self.acceptance_fraction
                logger.info(
                    f"  step {n}: acceptance fraction: mean {af.mean():.3f}, std {af.std():.3f}, "
                    f"min {af.min():.3f}, max {af.max():.3f}"
                )
        return result
