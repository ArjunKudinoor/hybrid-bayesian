"""Sampling implementation using pocoMC

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


def run_sampling(
    config: MCMCConfig,
    emulation_config: emulation.EmulationConfig,
    emulation_results: dict[str, dict[str, npt.NDArray[np.float64]]],
    emulator_cov_unexplained: dict,
    experimental_results: dict,
    parameter_min: npt.NDArray[np.float64],
    parameter_max: npt.NDArray[np.float64],
    parameter_ndim: int,
    closure_index: int,
    n_max_steps: int = -1,
) -> None:
    """Run with pocoMC.

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
    # pool = None

    # pocoMC config
    pocoMC_config = PocoMCConfig(
        analysis_name=config.analysis_name,
        parameterization=config.parameterization,
        analysis_config=config.analysis_config,
        config_file=config.config_file,
    )

    # Setup the prior distributions
    logging.info("Generate the prior class for pocoMC ...")
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
        logging.info("Starting pocoMC ...")
        sampler = pmc.Sampler(
            prior=prior,
            # likelihood=self.log_likelihood,
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
            pool=pool,
        )
        sampler.run(n_total=pocoMC_config.n_total_samples, n_evidence=pocoMC_config.n_importance_samples_for_evidence)

    logging.info("Generate the posterior samples ...")
    samples, weights, logl, logp = sampler.posterior()  # Weighted posterior samples

    logging.info("Generate the evidence ...")
    logz, logz_err = sampler.evidence()  # Bayesian model evidence estimate and uncertainty
    logger.info(f"Log evidence: {logz}")
    logger.info(f"Log evidence error: {logz_err}")

    logging.info("Writing pocoMC chains to file...")
    chain_data = {"chain": samples, "weights": weights, "logl": logl, "logp": logp, "logz": logz, "logz_err": logz_err}
    with config.mcmc_outputfile.open("wb") as file:
        pickle.dump(chain_data, file)


class PocoMCConfig(common_base.CommonBase):
    """Configuration class for pocoMC MCMC sampler."""

    def __init__(
        self, analysis_name="", parameterization="", analysis_config="", config_file="", closure_index=-1, **kwargs
    ):
        self.analysis_name = analysis_name
        self.parameterization = parameterization
        self.analysis_config = analysis_config
        self.config_file = Path(config_file)

        with self.config_file.open() as stream:
            config = yaml.safe_load(stream)

        self.observable_table_dir = config["observable_table_dir"]
        self.observable_config_dir = config["observable_config_dir"]
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
        # self.n_effective = mcmc_configuration.get("n_effective", 1000)
        # 512 is the default from pocoMC
        self.n_effective = mcmc_configuration.get("n_effective", 512)
        # n_active (int): The number of active particles (default is n_active=250). It must be smaller than n_ess.
        self.n_active = mcmc_configuration.get("n_active", 250)
        # Validation
        if self.n_active >= self.n_effective:
            msg = f"n_active ({self.n_active}) must be smaller than n_effective ({self.n_effective})."
            raise ValueError(msg)

        # n_prior (int): Number of prior samples to draw (default is n_prior=2*(n_effective//n_active)*n_active).
        self.draw_n_prior_samples = mcmc_configuration.get(
            "draw_n_prior_samples", 2 * (self.n_effective // self.n_active) * self.n_active
        )
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

        self.output_dir = Path(config["output_dir"]) / f"{analysis_name}_{parameterization}"
        self.emulation_outputfile = Path(self.output_dir) / "emulation.pkl"
        self.mcmc_outputfilename = "mcmc.h5"
        if closure_index < 0:
            self.mcmc_output_dir = Path(self.output_dir)
        else:
            self.mcmc_output_dir = Path(self.output_dir) / f"closure/results/{closure_index}"
        self.mcmc_outputfile = Path(self.mcmc_output_dir) / "mcmc.h5"
        self.sampler_outputfile = Path(self.mcmc_output_dir) / "mcmc_sampler.pkl"

        # Update formatting of parameter names for plotting
        unformatted_names = self.analysis_config["parameterization"][self.parameterization]["names"]
        self.analysis_config["parameterization"][self.parameterization]["names"] = [rf"{s}" for s in unformatted_names]
