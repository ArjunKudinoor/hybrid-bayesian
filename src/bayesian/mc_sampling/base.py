"""Base sampling functionality to compute the posterior.

The main functionalities are:
 - run_mcmc() performs MCMC and returns posterior
 - credible_interval() compute credible interval for a given posterior

A configuration class MCConfig provides simple access to MCMC settings.

# Users

Sampling is configured via MCConfig. The concept is that you configure a sampler
(e.g. emcee or pocoMC) which will run the MCMC. Use MCConfig.from_config_file()
to construct from the analysis YAML configuration, then call run_mcmc(config).

# Developers adding new samplers

To add a new sampler backend, implement the following in a new module in this directory:

- _register_name: str — name under which the sampler is registered
- SamplerSettings class — attrs class satisfying the SamplerSettings Protocol
- run_sampling(config, emulation_config, emulation_results, experimental_results,
               parameter_min, parameter_max, parameter_ndim) -> None

The framework will automatically discover and register modules with _register_name.
See emcee.py for the canonical implementation.

Based in part on JETSCAPE/STAT code.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Protocol, runtime_checkable

import attrs
import numpy as np
import numpy.typing as npt

from bayesian import analysis, data_IO, emulation, register_modules

logger = logging.getLogger(__name__)

_samplers: dict[str, ModuleType] = {}


@attrs.define
class BaseSamplerSettings:
    """Base (shared) settings for a sampler.

    Store this class in your specialized sampler settings class via composition.

    Attributes:
        settings: Dict from the YAML config corresponding to the MCMC settings.
    """

    settings: dict[str, Any] = attrs.field()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseSamplerSettings:
        return cls(settings=config)


@runtime_checkable
class SamplerSettings(Protocol):
    """Protocol for sampler settings classes.

    Attributes:
        sampler_name: Name of the sampler. Must match the module's _register_name.
        base_settings: Base sampler settings shared across sampler types.
        settings: Dictionary containing the full MCMC configuration.
    """

    sampler_name: ClassVar[str]
    base_settings: BaseSamplerSettings
    settings: dict[str, Any]


@attrs.define
class MCConfig:
    """Configuration for MC sampling.

    Attributes:
        analysis_settings: Analysis settings (determines paths, parameterization, etc.).
        sampler_settings: Sampler-specific settings (e.g. EmceeSamplerSettings).
        closure_index: If >= 0, run a closure test using this validation design point index.
    """

    analysis_settings: analysis.AnalysisSettings = attrs.field()
    sampler_settings: SamplerSettings = attrs.field()
    closure_index: int = attrs.field(default=-1)

    @classmethod
    def from_config_file(
        cls,
        analysis_settings: analysis.AnalysisSettings,
        closure_index: int = -1,
    ) -> MCConfig:
        """Initialize from an analysis settings object (reads MCMC config from YAML).

        Args:
            analysis_settings: Analysis settings object.
            closure_index: Closure test index. Default: -1 (no closure test).
        """
        mcmc_config = analysis_settings.raw_analysis_config["parameters"]["mcmc"]
        sampler_name = mcmc_config.get("mcmc_package", "emcee")
        try:
            sampler_module = _samplers[sampler_name]
        except KeyError as e:
            msg = f"Sampler backend '{sampler_name}' not registered or available. Available: {list(_samplers)}"
            raise KeyError(msg) from e
        sampler_settings = sampler_module.SamplerSettings.from_config(mcmc_config)
        return cls(
            analysis_settings=analysis_settings,
            sampler_settings=sampler_settings,
            closure_index=closure_index,
        )

    @property
    def output_dir(self) -> Path:
        return self.analysis_settings.output_dir

    @property
    def mcmc_output_dir(self) -> Path:
        if self.closure_index < 0:
            return self.output_dir
        return self.output_dir / f"closure/results/{self.closure_index}"

    @property
    def mcmc_outputfile(self) -> Path:
        return self.mcmc_output_dir / "mcmc.h5"

    @property
    def mcmc_outputfilename(self) -> str:
        return self.mcmc_outputfile.name

    @property
    def sampler_outputfile(self) -> Path:
        return self.mcmc_output_dir / "mcmc_sampler.pkl"

    # Convenience accessors — mirror old MCMCConfig attributes used by plot files
    @property
    def analysis_name(self) -> str:
        return self.analysis_settings.name

    @property
    def parameterization(self) -> str:
        return self.analysis_settings.parameterization

    @property
    def analysis_config(self) -> dict[str, Any]:
        return self.analysis_settings.raw_analysis_config

    @property
    def config_file(self) -> Path:
        return self.analysis_settings.config_file

    @property
    def observables_filename(self) -> str:
        return self.analysis_settings.io.observables_filename

    @property
    def observable_table_dir(self) -> Path:
        return self.analysis_settings.io.observables_table_dir

    @property
    def confidence(self) -> float:
        return float(self.analysis_settings.raw_analysis_config.get("closure", {}).get("confidence", 0.9))


def run_mcmc(config: MCConfig) -> None:
    """Run MCMC to compute the posterior.

    Loads emulators and experimental data, then dispatches to the configured
    sampler plugin (e.g. emcee or pocoMC).

    Args:
        config: MC sampling configuration.
    """
    parameterization = config.analysis_settings.parameterization
    param_cfg = config.analysis_settings.raw_analysis_config["parameterization"][parameterization]
    parameter_min = np.array(param_cfg["min"])
    parameter_max = np.array(param_cfg["max"])
    ndim = len(param_cfg["names"])

    # Load emulators from disk
    emulation_config = emulation.EmulationConfig.from_config_file(analysis_settings=config.analysis_settings)
    emulation_results = emulation_config.read_all_emulator_groups(config.analysis_settings)

    # Load experimental data. For closure tests, pseudodata replaces real data.
    experimental_results = data_IO.data_array_from_h5(
        config.output_dir,
        "observables.h5",
        pseudodata_index=config.closure_index,
        observable_filter=emulation_config.observable_filter,
    )

    sampler_name = config.sampler_settings.sampler_name
    try:
        sampler = _samplers[sampler_name]
    except KeyError as e:
        msg = f"Sampler '{sampler_name}' not found in registry. Available: {list(_samplers)}"
        raise KeyError(msg) from e

    sampler.run_sampling(
        config=config,
        emulation_config=emulation_config,
        emulation_results=emulation_results,
        experimental_results=experimental_results,
        parameter_min=parameter_min,
        parameter_max=parameter_max,
        parameter_ndim=ndim,
    )


def credible_interval(
    samples: npt.NDArray[np.float64], confidence: float = 0.9, interval_type: str = "quantile"
) -> tuple[float, float]:
    """Compute the credible interval for an array of samples.

    Args:
        samples: Array of samples.
        confidence: Confidence level. Default: 0.9.
        interval_type: 'hpd' (highest posterior density) or 'quantile'. Default: 'quantile'.
    """
    if interval_type == "hpd":
        nci = int((1 - confidence) * samples.size)
        argp = np.argpartition(samples, [nci, samples.size - nci])
        cil = np.sort(samples[argp[:nci]])
        cih = np.sort(samples[argp[-nci:]])
        ihpd = np.argmin(cih - cil)
        ci = cil[ihpd], cih[ihpd]
    elif interval_type == "quantile":
        cred_range = [(1 - confidence) / 2, 1 - (1 - confidence) / 2]
        ci = np.quantile(samples, cred_range)  # type: ignore[assignment]
    return ci


def map_parameters(posterior: npt.NDArray[np.float64], method: str = "quantile") -> npt.NDArray[np.float64]:
    """Compute the MAP parameters.

    Args:
        posterior: Array of posterior samples, shape (n_samples, n_parameters).
        method: 'quantile' — narrow quantile interval mean. Default: 'quantile'.
    """
    if method == "quantile":
        central_quantile = 0.01
        lower_bounds = np.quantile(posterior, 0.5 - central_quantile / 2, axis=0)
        upper_bounds = np.quantile(posterior, 0.5 + central_quantile / 2, axis=0)
        mask = (posterior >= lower_bounds) & (posterior <= upper_bounds)
        return np.array([posterior[mask[:, i], i].mean() for i in range(posterior.shape[1])])
    msg = f"Unknown method: {method}"
    raise ValueError(msg)


def _validate_sampler(name: str, module: ModuleType) -> None:
    """Validate that a sampler module follows the expected interface."""
    for attr in ["run_sampling", "SamplerSettings"]:
        if not hasattr(module, attr):
            msg = f"Sampler module '{name}' is missing required attribute '{attr}'"
            raise ValueError(msg)


# Discover and register sampler plugin modules at import time
if not _samplers:
    _samplers.update(
        register_modules.discover_and_register_modules(
            calling_module_name=__name__,
            required_attributes=["SamplerSettings", "run_sampling"],
            validation_function=_validate_sampler,
        )
    )
