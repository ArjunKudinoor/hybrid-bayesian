"""Base sampling functionality to compute the posterior.

The main functionalities are:
 - run_mcmc() performs MCMC and returns posterior
 - credible_interval() compute credible interval for a given posterior

A configuration class MCMCConfig provides simple access to emulation settings

Based in part on JETSCAPE/STAT code.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml

from bayesian import common_base, data_IO, emulation, register_modules

logger = logging.getLogger(__name__)

_samplers: dict[str, ModuleType] = {}


####################################################################################################################
def run_mcmc(config: MCMCConfig, closure_index: int = -1) -> None:
    """
    Run MCMC to compute posterior

    :param MCMCConfig config: Instance of MCMCConfig
    :param int closure_index: Index of validation design point to use for MCMC closure. Off by default.
                              If non-negative index is specified, will construct pseudodata from the design point
                              and use that for the closure test.
    """

    # Get parameter names and min/max
    names = config.analysis_config["parameterization"][config.parameterization]["names"]
    parameter_min = config.analysis_config["parameterization"][config.parameterization]["min"]
    parameter_max = config.analysis_config["parameterization"][config.parameterization]["max"]
    ndim = len(names)

    # Load emulators
    emulation_config = emulation.EmulationConfig.from_config_file(
        analysis_settings=config.analysis_config,
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
    experimental_results = data_IO.data_array_from_h5(
        config.output_dir,
        "observables.h5",
        pseudodata_index=closure_index,
        observable_filter=emulation_config.observable_filter,
    )

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
def credible_interval(samples, confidence=0.9, interval_type="quantile"):
    """
    Compute the credible interval for an array of samples.

    TODO: one could also call the versions in pymc3 or arviz

    :param 1darray samples: Array of samples
    :param float confidence: Confidence level (default 0.9)
    :param str type: Type of credible interval to compute. Options are:
                        'hpd' - highest-posterior density
                        'quantile' - quantile interval
    """

    if interval_type == "hpd":
        # number of intervals to compute
        nci = int((1 - confidence) * samples.size)
        # find highest posterior density (HPD) credible interval i.e. the one with minimum width
        argp = np.argpartition(samples, [nci, samples.size - nci])
        cil = np.sort(samples[argp[:nci]])  # interval lows
        cih = np.sort(samples[argp[-nci:]])  # interval highs
        ihpd = np.argmin(cih - cil)
        ci = cil[ihpd], cih[ihpd]

    elif interval_type == "quantile":
        cred_range = [(1 - confidence) / 2, 1 - (1 - confidence) / 2]
        ci = np.quantile(samples, cred_range)

    return ci


####################################################################################################################
def map_parameters(posterior, method="quantile"):
    """
    Compute the MAP parameters

    :param 1darray posterior: Array of samples
    :param str method: Method used to compute MAP. Options are:
                        'quantile' - take a narrow quantile interval and compute mean of parameters in that interval
    :return 1darray map_parameters: Array of MAP parameters
    """

    if method == "quantile":
        central_quantile = 0.01
        lower_bounds = np.quantile(posterior, 0.5 - central_quantile / 2, axis=0)
        upper_bounds = np.quantile(posterior, 0.5 + central_quantile / 2, axis=0)
        mask = (posterior >= lower_bounds) & (posterior <= upper_bounds)
        map_parameters = np.array([posterior[mask[:, i], i].mean() for i in range(posterior.shape[1])])

    return map_parameters


def _validate_sampler(name: str, module: ModuleType) -> None:
    """
    Validate that an emulator module follows the expected interface.
    """
    if not hasattr(module, "fit_emulator"):
        msg = f"Emulator module {name} does not have a required 'fit_emulator' method"
        raise ValueError(msg)
    # TODO: Re-enable when things stabilize a bit.
    # if not hasattr(module, "predict"):
    #     msg = f"Emulator module {name} does not have a required 'predict' method"
    #     raise ValueError(msg)


class MCMCConfig(common_base.CommonBase):
    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
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

        mcmc_configuration = analysis_config["parameters"]["mcmc"]
        # General arguments
        self.mcmc_package = mcmc_configuration.get("mcmc_package", "emcee")
        # emcee specific
        self.n_walkers = mcmc_configuration["n_walkers"]
        self.n_burn_steps = mcmc_configuration["n_burn_steps"]
        self.n_sampling_steps = mcmc_configuration["n_sampling_steps"]
        self.n_logging_steps = mcmc_configuration["n_logging_steps"]

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


# Actually perform the discovery and registration of the emulators
if not _samplers:
    _samplers.update(
        register_modules.discover_and_register_modules(
            calling_module_name=__name__,
            required_attributes=[],
            validation_function=_validate_sampler,
        )
    )
