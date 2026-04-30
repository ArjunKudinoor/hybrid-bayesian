"""Sampling implementation using pocoMC

Implementation of the mc_sampling plugin interface using the Preconditioned Monte Carlo
(PMC) sampler from pocoMC. PMC uses normalizing flows to precondition the target
distribution for more efficient sampling.

This implementation draws heavily on the wrapper by Hendrik Roch:
https://github.com/Hendrik1704/GPBayesTools-HIC

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
from typing import Any, ClassVar

import attrs
import numpy as np
import numpy.typing as npt

from bayesian import emulation, log_posterior
from bayesian.mc_sampling import base as mc_sampling_base

logger = logging.getLogger(__name__)

_register_name = "pocoMC"


@attrs.define
class SamplerSettings:
    """Settings for the pocoMC Preconditioned Monte Carlo sampler.

    Attributes:
        sampler_name: Name matching _register_name.
        base_settings: Base sampler settings (holds raw config dict).
        n_effective: Effective sample size maintained during the run (default: 512).
        n_active: Number of active particles; must be < n_effective (default: 250).
        draw_n_prior_samples: Number of prior samples to draw initially.
        sampler_type: MCMC kernel type — 'tpcn' (t-preconditioned Crank-Nicolson,
            recommended) or 'rwm' (random-walk Metropolis). Default: 'tpcn'.
        n_total_samples: Total effectively independent samples to collect (default: 5000).
        n_importance_samples_for_evidence: Importance samples for evidence estimation
            (default: 5000). Set to 0 to use SMC estimate instead.
        settings: Raw MCMC config dict.
    """

    sampler_name: ClassVar[str] = "pocoMC"
    base_settings: mc_sampling_base.BaseSamplerSettings = attrs.field()
    n_effective: int = attrs.field()
    n_active: int = attrs.field()
    draw_n_prior_samples: int = attrs.field()
    sampler_type: str = attrs.field()
    n_total_samples: int = attrs.field()
    n_importance_samples_for_evidence: int = attrs.field()
    settings: dict[str, Any] = attrs.field()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SamplerSettings:
        poco = config.get("pocoMC", {})
        n_effective = poco.get("n_effective", 512)
        n_active = poco.get("n_active", 250)
        if n_active >= n_effective:
            msg = f"n_active ({n_active}) must be smaller than n_effective ({n_effective})"
            raise ValueError(msg)
        return cls(
            base_settings=mc_sampling_base.BaseSamplerSettings.from_config(config),
            n_effective=n_effective,
            n_active=n_active,
            draw_n_prior_samples=poco.get("draw_n_prior_samples", 2 * (n_effective // n_active) * n_active),
            sampler_type=poco.get("sampler_type", "tpcn"),
            n_total_samples=poco.get("n_total_samples", 5000),
            n_importance_samples_for_evidence=poco.get("n_importance_samples_for_evidence", 5000),
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
    """Run pocoMC-based Preconditioned Monte Carlo sampling.

    Args:
        config: MC sampling configuration (includes output paths, closure_index).
        emulation_config: Emulation configuration.
        emulation_results: Trained emulator results, keyed by emulator group name.
        experimental_results: Experimental data arrays.
        parameter_min: Lower bounds for each parameter.
        parameter_max: Upper bounds for each parameter.
        parameter_ndim: Number of parameters.
    """
    import pocomc as pmc  # noqa: PLC0415
    import scipy.stats  # noqa: PLC0415

    sampler_settings: SamplerSettings = config.sampler_settings

    n_max_steps = 10 * parameter_ndim

    # Build uniform prior distributions over parameter bounds
    logger.info("Constructing prior distributions for pocoMC...")
    prior_distributions = [
        scipy.stats.uniform(p_min, p_max - p_min) for p_min, p_max in zip(parameter_min, parameter_max, strict=True)
    ]
    prior = pmc.Prior(prior_distributions)

    # NOTE: We need to use `spawn` rather than `fork` on linux. Otherwise, some caching
    #       mechanisms (e.g. used in learning the emulator group mapping) don't work.
    # NOTE: We create the pool manually (rather than using pocoMC's built-in) so that we
    #       can initialize log_posterior globals in each worker process.
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
        logger.info("Starting pocoMC sampler...")
        sampler = pmc.Sampler(
            prior=prior,
            likelihood=log_posterior.log_posterior,
            likelihood_kwargs={"set_to_infinite_outside_bounds": False},
            n_effective=sampler_settings.n_effective,
            n_active=sampler_settings.n_active,
            n_prior=sampler_settings.draw_n_prior_samples,
            sample=sampler_settings.sampler_type,
            n_max_steps=n_max_steps,
            vectorize=True,
            pool=pool,
        )
        sampler.run(
            n_total=sampler_settings.n_total_samples,
            n_evidence=sampler_settings.n_importance_samples_for_evidence,
        )

    logger.info("Extracting posterior samples...")
    samples, weights, logl, logp = sampler.posterior()

    logger.info("Estimating Bayesian evidence...")
    logz, logz_err = sampler.evidence()
    logger.info(f"Log evidence: {logz:.4f} +/- {logz_err:.4f}")

    logger.info("Writing pocoMC results to file...")
    chain_data = {
        "chain": samples,
        "weights": weights,
        "logl": logl,
        "logp": logp,
        "logz": logz,
        "logz_err": logz_err,
    }
    config.mcmc_output_dir.mkdir(exist_ok=True, parents=True)
    with config.mcmc_outputfile.open("wb") as f:
        pickle.dump(chain_data, f)

    logger.info("Done.")
