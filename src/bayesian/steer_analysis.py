"""
Main script to steer Bayesian inference studies for heavy-ion jet analysis

authors: J.Mulligan, R.Ehlers
Based in part on JETSCAPE/STAT code.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import yaml

from bayesian import (
    analysis,
    common_base,
    data_IO,
    emulation,
    helpers,
    plot_analyses,
    plot_closure,
    plot_covariance,
    plot_emulation,
    plot_input_data,
    plot_mcmc,
    plot_qhat,
    preprocess_input_data,
)
from bayesian import mc_sampling as mcmc

logger = logging.getLogger(__name__)


####################################################################################################################
class SteerAnalysis(common_base.CommonBase):
    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, config_file: Path, **kwargs):
        # Initialize config file
        self.config_file = config_file
        self.initialize()

        logger.info(self)

    # ---------------------------------------------------------------
    # Initialize config
    # ---------------------------------------------------------------
    def initialize(self) -> None:
        logger.info("Initializing class objects")

        with self.config_file.open() as stream:
            config = yaml.safe_load(stream)

        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Option to reduce logging to file
        self._reduce_logging_to_file = config.get("reduce_logging_to_file", False)

        # Data inputs
        self.observable_table_dir = config["observable_table_dir"]
        self.observable_config_dir = config["observable_config_dir"]

        # Configure which functions to run
        self.initialize_observables = config["initialize_observables"]
        self.preprocess_input_data = config["preprocess_input_data"]
        self.fit_emulators = config["fit_emulators"]
        self.run_mcmc = config["run_mcmc"]
        self.run_closure_tests = config["run_closure_tests"]
        self.plot = config["plot"]

        # Configuration of different analyses
        all_analyses_config = config["analyses"]
        self.correlation_groups = all_analyses_config.pop("correlation_groups", {})
        self.analyses = all_analyses_config  # Now only contains actual analyses

        if self.correlation_groups:
            logger.info(f"Loaded correlation_groups with {len(self.correlation_groups)} group tags")

    def run_analysis(self) -> None:
        """Main steering function for analyses."""
        # Keep track of log and config for each run for reproducibility.
        if not self._reduce_logging_to_file:
            # Add logging to file
            _root_log = logging.getLogger()
            _root_log.addHandler(logging.FileHandler(self.output_dir / "steer_analysis.log", "w"))

            # Also write analysis config to shared directory
            shutil.copy(self.config_file, Path(self.output_dir) / "steer_analysis_config.yaml")

        # Loop through each analysis
        with helpers.progress_bar() as progress:
            analysis_task = progress.add_task("[deep_sky_blue1]Running analysis...", total=len(self.analyses))

            for analysis_name, analysis_config in self.analyses.items():
                # Now you don't need the skip check anymore!

                # Loop through the parameterizations
                parameterization_task = progress.add_task(
                    "[deep_sky_blue2]parameterization", total=len(analysis_config.get("parameterizations", ["default"]))
                )

            for analysis_name, analysis_config in self.analyses.items():
                if analysis_name == "correlation_groups":  # Skip special config keys
                    continue

                # Loop through the parameterizations
                parameterization_task = progress.add_task(
                    "[deep_sky_blue2]parameterization", total=len(analysis_config.get("parameterizations", ["default"]))
                )

                for parameterization in analysis_config.get("parameterizations", ["default"]):
                    analysis_settings = analysis.AnalysisSettings.from_config_file(
                        analysis_name=analysis_name,
                        parameterization=parameterization,
                        config_file=self.config_file,
                    )

                    # Initialize design points, predictions, data, and uncertainties
                    # We store them in a dict and write/read it to HDF5
                    if self.initialize_observables:
                        # Just indicate that it's working
                        initialization_task = progress.add_task("[deep_sky_blue4]Initializing...", total=None)
                        progress.start_task(initialization_task)
                        logger.info("")
                        logger.info("========================================================================")
                        logger.info(f"Initializing model: {analysis_name} ({parameterization} parameterization)...")

                        observables = data_IO.initialize_observables_dict_from_tables(
                            self.observable_table_dir,
                            analysis_config,
                            parameterization,
                            correlation_groups=self.correlation_groups,
                        )
                        data_IO.write_dict_to_h5(
                            observables,
                            os.path.join(self.output_dir, f"{analysis_name}_{parameterization}"),
                            filename="observables.h5",
                        )
                        progress.update(initialization_task, advance=100, visible=False)

                    output_dir = os.path.join(self.output_dir, f"{analysis_name}_{parameterization}")
                    experimental_data = data_IO.data_array_from_h5(output_dir, "observables.h5")

                    if "external_covariance" in experimental_data:
                        ext_cov = experimental_data["external_covariance"]

                    if self.preprocess_input_data:
                        # Just indicate that it's working
                        preprocess_task = progress.add_task("[deep_sky_blue4]Preprocessing...", total=None)
                        progress.start_task(preprocess_task)
                        logger.info("")
                        logger.info("------------------------------------------------------------------------")
                        logger.info(
                            f"Preprocessing input data: {analysis_name} ({parameterization} parameterization)..."
                        )

                        preprocessing_config = preprocess_input_data.PreprocessingConfig(
                            analysis_name=analysis_name,
                            parameterization=parameterization,
                            analysis_config=analysis_config,
                            config_file=self.config_file,
                        )
                        # NOTE: Strictly speaking, we don't want the emulation config here. However,
                        #       We often need the observable filter, and it doesn't cost anything to
                        #       construct here, so we just go for it.
                        # emulation_config = emulation.EmulationConfig.from_config_file(
                        #    analysis_name=analysis_name,
                        #    parameterization=parameterization,
                        #    analysis_config=analysis_config,
                        #    config_file=self.config_file,
                        # )
                        observables_smoothed = preprocess_input_data.preprocess(
                            preprocessing_config=preprocessing_config,
                        )
                        data_IO.write_dict_to_h5(
                            observables_smoothed,
                            os.path.join(self.output_dir, f"{analysis_name}_{parameterization}"),
                            filename="observables_preprocessed.h5",
                        )
                        progress.update(preprocess_task, advance=100, visible=False)

                    # Fit emulators and write them to file
                    if self.fit_emulators:
                        # Just indicate that it's working
                        emulation_task = progress.add_task("[deep_sky_blue4]Emulating...", total=None)
                        progress.start_task(emulation_task)
                        logger.info("------------------------------------------------------------------------")
                        logger.info(f"Fitting emulators for {analysis_name}_{parameterization}...")
                        emulation_config = emulation.EmulationConfig.from_config_file(
                            analysis_name=analysis_name,
                            parameterization=parameterization,
                            analysis_config=analysis_config,
                            config_file=self.config_file,
                        )
                        emulation.fit_emulators(emulation_config)
                        progress.update(emulation_task, advance=100, visible=False)

                    # Run MCMC
                    if self.run_mcmc:
                        # Just indicate that it's working
                        mcmc_task = progress.add_task("[deep_sky_blue4]Running MCMC...", total=None)
                        progress.start_task(mcmc_task)
                        logger.info("")
                        logger.info("------------------------------------------------------------------------")
                        logger.info(f"Running MCMC for {analysis_name}_{parameterization}...")
                        mcmc_config = mcmc.MCConfig.from_config_file(
                            analysis_settings=analysis_settings,
                        )
                        mcmc.run_mcmc(mcmc_config)
                        progress.update(mcmc_task, advance=100, visible=False)

                    # Run closure tests -- one for each validation design point
                    #   - Use validation point as pseudodata
                    #   - Use emulator already trained on training points
                    if self.run_closure_tests:
                        validation_indices = list(
                            range(analysis_config["validation_indices"][0], analysis_config["validation_indices"][1])
                        )
                        n_design_points = len(validation_indices)
                        closure_test_task = progress.add_task(
                            "[deep_sky_blue4]Running closure tests...", total=n_design_points
                        )
                        progress.start_task(closure_test_task)
                        logger.info("")
                        logger.info("------------------------------------------------------------------------")

                        for i, validation_design_point in enumerate(validation_indices):
                            logger.info(
                                f"Running closure tests for {analysis_name}_{parameterization}, validation_design_point={validation_design_point}, validation_index={i}..."
                            )
                            mcmc_config = mcmc.MCConfig.from_config_file(
                                analysis_settings=analysis_settings,
                                closure_index=i,
                            )  # Use validation array index, not design point ID
                            mcmc.run_mcmc(mcmc_config)
                            progress.update(closure_test_task, advance=1)

                    # progress.update(parameterization_task, advance=1)
                # Hide once we're done!
                progress.update(parameterization_task, visible=False)

                progress.update(analysis_task, advance=1)

        # Plots for individual analysis
        for analysis_name, analysis_config in self.analyses.items():
            for parameterization in analysis_config.get("parameterizations", ["default"]):
                if any(self.plot.values()):
                    logger.info("========================================================================")
                    logger.info(f"Plotting for {analysis_name} ({parameterization} parameterization)...")
                    logger.info("")

                if self.plot["input_data"]:
                    logger.info("------------------------------------------------------------------------")
                    logger.info(f"Plotting input data for {analysis_name}_{parameterization}...")
                    emulation_config = emulation.EmulationConfig.from_config_file(
                        analysis_name=analysis_name,
                        parameterization=parameterization,
                        analysis_config=analysis_config,
                        config_file=self.config_file,
                    )
                    plot_input_data.plot(emulation_config)
                    logger.info("Done!")
                    logger.info("")

                if self.plot["emulators"]:
                    logger.info("------------------------------------------------------------------------")
                    logger.info(f"Plotting emulators for {analysis_name}_{parameterization}...")
                    emulation_config = emulation.EmulationConfig.from_config_file(
                        analysis_name=analysis_name,
                        parameterization=parameterization,
                        analysis_config=analysis_config,
                        config_file=self.config_file,
                    )
                    plot_emulation.plot(emulation_config)
                    logger.info("Done!")
                    logger.info("")

                if self.plot["mcmc"] or self.plot["qhat"] or self.plot["closure_tests"]:
                    _plot_analysis_settings = analysis.AnalysisSettings.from_config_file(
                        analysis_name=analysis_name,
                        config_file=self.config_file,
                        parameterization=parameterization,
                    )
                    _plot_mcmc_config = mcmc.MCConfig.from_config_file(
                        analysis_settings=_plot_analysis_settings,
                    )

                if self.plot["mcmc"]:
                    logger.info("------------------------------------------------------------------------")
                    logger.info(f"Plotting MCMC for {analysis_name}_{parameterization}...")
                    plot_mcmc.plot(_plot_mcmc_config)
                    logger.info("Done!")
                    logger.info("")

                if self.plot["covariance"]:
                    logger.info("------------------------------------------------------------------------")
                    logger.info(f"Plotting covariance matrices for {analysis_name}_{parameterization}...")
                    plot_covariance.plot(analysis_name, parameterization, analysis_config, self.config_file)
                    logger.info("Done!")
                    logger.info("")

                if self.plot["qhat"]:
                    logger.info("------------------------------------------------------------------------")
                    logger.info(f"Plotting qhat results {analysis_name}_{parameterization}...")
                    plot_qhat.plot(_plot_mcmc_config)
                    logger.info("Done!")
                    logger.info("")

                if self.plot["closure_tests"]:
                    logger.info("------------------------------------------------------------------------")
                    logger.info(f"Plotting closure test results {analysis_name}_{parameterization}...")
                    plot_closure.plot(_plot_mcmc_config)
                    logger.info("Done!")
                    logger.info("")

        # Plots across multiple analyses
        if self.plot["across_analyses"]:
            # NOTE: This is a departure from the standard API, but we need a convention for how
            #       to pass multiple analyses, so we'll just go with it for now.
            plot_analyses.plot(self.analyses, self.config_file, self.output_dir)


def main() -> None:
    helpers.setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Jet Bayesian Analysis")
    parser.add_argument(
        "-c",
        "--configFile",
        help="Path of config file for analysis",
        action="store",
        type=Path,
        default=Path("../config/jet_substructure.yaml"),
    )
    args = parser.parse_args()

    logger.info("Configuring...")
    logger.info(f"  configFile: {args.configFile}")

    # If invalid configFile is given, exit
    config_file = Path(args.configFile)
    if not config_file.exists():
        msg = f"File {args.configFile} does not exist! Exiting!"
        logger.info(msg)
        raise ValueError(msg)

    steer_analysis = SteerAnalysis(config_file=config_file)
    steer_analysis.run_analysis()


if __name__ == "__main__":
    main()
