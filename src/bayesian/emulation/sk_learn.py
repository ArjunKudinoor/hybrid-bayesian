"""Gaussian Process Emulator from scikit-learn.

This emulator does a PCA on the data and truncates at some number of components
before fitting to reduce the dimensionality of the training. As of Nov 2025,
this is common in heavy-ions, and seems to provide reasonable performance.
Note that this truncation leads to some additional covariance term, which we
track and propagate.

Based in part on JETSCAPE/STAT code.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
.. codeauthor:: James Mulligan, LBL/UCB
.. codeauthor:: Jingyu Zhang <jingyu.zhang@cern.ch>, Vanderbilt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import attrs
import numpy as np
import numpy.typing as npt
import sklearn.decomposition as sklearn_decomposition  # type: ignore[import-untyped]
import sklearn.gaussian_process as sklearn_gaussian_process
import sklearn.preprocessing as sklearn_preprocessing
import yaml

from bayesian import analysis, data_IO
from bayesian.emulation import base as emulation_base

logger = logging.getLogger(__name__)

# Name under which the module is registered.
_register_name = "sk_learn"


def fit_emulator(
    emulator_settings: SKLearnEmulatorSettings, analysis_settings: analysis.AnalysisSettings
) -> dict[str, Any]:
    """Do PCA and fit the emulator.

    The first config.n_pc principal components (PCs) are emulated by independent Gaussian processes (GPs)
    The emulators map design points to PCs; the output will need to be inverted from PCA space to physical space.

    Args:

        config: we take an instance of (particular) EmulationConfig as an argument to keep track of config info.
        analysis_settings: Analysis settings
    """
    # Setup
    output_filename = emulation_base.IO.output_filename(
        emulator_settings=emulator_settings, analysis_settings=analysis_settings
    )

    # Check if emulator already exists
    if output_filename.exists():
        if emulator_settings.force_retrain:
            output_filename.unlink()
            logger.info(f"Removed {output_filename}")
        else:
            logger.info(f"Emulators already exist: {output_filename} (to force retrain, set force_retrain: True)")
            return {}

    # Initialize predictions into a single 2D array: (design_point_index, observable_bins) i.e. (n_samples, n_features)
    # A consistent order of observables is enforced internally in data_IO
    # NOTE: One sample corresponds to one design point, while one feature is one bin of one observable
    logger.info("Doing PCA...")
    Y = data_IO.predictions_matrix_from_h5(
        output_dir=analysis_settings.output_dir,
        filename=analysis_settings.io.observables_filename,
        observable_filter=emulator_settings.base_settings.observable_filter,
    )

    # Use sklearn to:
    #  - Center and scale each feature (and later invert)
    #  - Perform PCA to reduce to config.n_pc features.
    #      This amounts to finding the matrix S that diagonalizes the covariance matrix C = Y.T*Y = S*D^2*S.T
    #      Or equivalently the right singular vectors S.T in the SVD decomposition of Y: Y = U*D*S.T
    #      Given S, we can transform from feature space to PCA space with: Y_PCA = Y*S
    #               and from PCA space to feature space with: Y = Y_PCA*S.T
    #
    # The input Y is a 2D array of format (n_samples, n_features).
    #
    # The output of pca.fit_transform() is a 2D array of format (n_samples, n_components),
    #   which is equivalent to:
    #     Y_pca = Y.dot(pca.components_.T), where:
    #       pca.components_ are the principal axes, sorted by decreasing explained variance -- shape (n_components, n_features)
    #     In the notation above, pca.components_ = S.T, i.e.
    #       the rows of pca.components_ are the sorted eigenvectors of the covariance matrix of the scaled features.
    #
    # We can invert this back to the original feature space by: pca.inverse_transform(Y_pca),
    #   which is equivalent to:
    #     Y_reconstructed = Y_pca.dot(pca.components_)
    #
    # Then, we still need to make sure to undo the preprocessing (centering and scaling) by:
    #     Y_reconstructed_unscaled = scaler.inverse_transform(Y_reconstructed)
    #
    # See docs for StandardScaler and PCA for further details.
    # This post explains exactly what fit_transform,inverse_transform do: https://stackoverflow.com/a/36567821
    #
    # TODO: Do we want whiten the PCs, i.e. to scale the variances of each PC to 1?
    #       I don't see a compelling reason to do this...We are fitting separate GPs to each PC,
    #       so standardizing the variance of each PC is not important.
    #       (NOTE: whitening can be done with whiten=True -- beware that inverse_transform also undoes whitening)
    scaler = sklearn_preprocessing.StandardScaler()
    # This adopts the sklearn convention, but then sets a max cap of 30 PCs (arbitrarily chosen) to
    # reduce computation time.
    max_n_components = emulator_settings.max_n_components_to_calculate
    if max_n_components is not None:
        logger.info(f"Running with max n_pc={max_n_components}")
    # NOTE-STAT: Whiten=True, but here, Whiten=False.
    # NOTE-STAT: RJE thinks this doesn't matter, based on the comments above.
    pca = sklearn_decomposition.PCA(
        n_components=max_n_components, svd_solver="full", whiten=False
    )  # Include all PCs here, so we can access them later
    # Scale data and perform PCA
    Y_pca = pca.fit_transform(scaler.fit_transform(Y))
    Y_pca_truncated = Y_pca[:, : emulator_settings.n_pc]  # Select PCs here
    # Invert PCA and undo the scaling
    Y_reconstructed_truncated = Y_pca_truncated.dot(pca.components_[: emulator_settings.n_pc, :])
    Y_reconstructed_truncated_unscaled = scaler.inverse_transform(Y_reconstructed_truncated)
    explained_variance_ratio = pca.explained_variance_ratio_
    logger.info(
        f"  Variance explained by first {emulator_settings.n_pc} components: {np.sum(explained_variance_ratio[: emulator_settings.n_pc])}"
    )

    # Get design
    design = data_IO.design_array_from_h5(
        analysis_settings.output_dir, filename=analysis_settings.io.observables_filename
    )

    # Define GP kernel (covariance function)
    min = np.array(analysis_settings.raw_analysis_config["parameterization"][analysis_settings.parameterization]["min"])
    max = np.array(analysis_settings.raw_analysis_config["parameterization"][analysis_settings.parameterization]["max"])

    kernel = None
    for kernel_type, kernel_args in emulator_settings.active_kernels.items():
        if kernel_type == "matern":
            length_scale = max - min
            length_scale_bounds_factor = kernel_args["length_scale_bounds_factor"]
            length_scale_bounds = np.outer(length_scale, tuple(length_scale_bounds_factor))
            nu = kernel_args["nu"]
            kernel = sklearn_gaussian_process.kernels.Matern(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds,
                nu=nu,
            )
        if kernel_type == "rbf":
            length_scale = max - min
            length_scale_bounds_factor = kernel_args["length_scale_bounds_factor"]
            length_scale_bounds = np.outer(length_scale, tuple(length_scale_bounds_factor))
            kernel = sklearn_gaussian_process.kernels.RBF(
                length_scale=length_scale, length_scale_bounds=length_scale_bounds
            )
        if kernel_type == "constant":
            constant_value = kernel_args["constant_value"]
            constant_value_bounds = kernel_args["constant_value_bounds"]
            kernel_constant = sklearn_gaussian_process.kernels.ConstantKernel(
                constant_value=constant_value, constant_value_bounds=constant_value_bounds
            )
            kernel = kernel + kernel_constant
        if kernel_type == "noise":
            kernel_noise = sklearn_gaussian_process.kernels.WhiteKernel(
                noise_level=kernel_args["args"]["noise_level"],
                noise_level_bounds=kernel_args["args"]["noise_level_bounds"],
            )
            kernel = kernel + kernel_noise

    # Fit a GP (optimize the kernel hyperparameters) to map each design point to each of its PCs
    # Note that Y_PCA=(n_samples, n_components), so each PC corresponds to a row (i.e. a column of Y_PCA.T)
    logger.info("")
    logger.info("Fitting GPs...")
    logger.info(f"  The design has {design.shape[1]} parameters")
    emulators = [
        sklearn_gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            alpha=emulator_settings.alpha,
            n_restarts_optimizer=emulator_settings.n_restarts,
            copy_X_train=False,
        ).fit(design, y)
        for y in Y_pca_truncated.T
    ]

    # Print hyperparameters.
    logger.info("")
    logger.info("Kernel hyperparameters:")
    [logger.info(f"  {emulator.kernel_}") for emulator in emulators]  # type: ignore[func-returns-value]
    logger.info("")

    # Write all info we want to the output dictionary.
    output_dict: dict[str, Any] = {}
    output_dict["PCA"] = {}
    output_dict["PCA"]["Y"] = Y
    output_dict["PCA"]["Y_pca"] = Y_pca
    output_dict["PCA"]["Y_pca_truncated"] = Y_pca_truncated
    output_dict["PCA"]["Y_reconstructed_truncated"] = Y_reconstructed_truncated
    output_dict["PCA"]["Y_reconstructed_truncated_unscaled"] = Y_reconstructed_truncated_unscaled
    output_dict["PCA"]["pca"] = pca
    output_dict["PCA"]["scaler"] = scaler
    output_dict["emulators"] = emulators

    return output_dict


def predict(
    parameters: npt.NDArray[np.float64],
    results: dict[str, Any],
    emulator_settings: EmulatorSettings,
    additional_covariance: npt.NDArray[np.float64] | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Predict the values at the given parameters by calculating their expected value via the emulator.

    This function generally implements predict for a set of emulators where we do PCA beforehand.
    However, enough of the details are specific to the sk_learn implementation, such that we can't
    use it fully generically.

    NOTE:
        One can easily construct a dict of predictions with format emulator_predictions[observable_label]
        from the returned matrix as follows (useful for plotting / troubleshooting):
        ```python
        observables = data_IO.read_dict_from_h5(config.output_dir, 'observables.h5', verbose=False)
        emulator_predictions = data_IO.observable_dict_from_matrix(
            emulator_central_value_reconstructed,
            observables,
            cov=emulator_cov_reconstructed,
            validation_set=validation_set
        )
       ```

    Args:
        parameters: Array of parameter values (e.g. [tau0, c1, c2, ...]), with shape (n_samples, n_parameters).
        results: Dictionary that stores output from the emulator.
        emulator_settings: Emulator settings.
        additional_covariance: Addition to the covariance due to the emulator.

    Returns:
        emulator_predictions: dictionary containing matrices of central values and covariance
    """

    # The emulators are stored as a list (one for each PC)
    emulators = results["emulators"]

    if additional_covariance is None:
        # Here, this corresponds to the unexplained covariance due to truncated the PCA.
        # See this function for additional details.
        additional_covariance = compute_additional_covariance_contributions(
            emulator_settings=emulator_settings,
            emulator_result=results,
        )

    # Get predictions (in PC space) from each emulator and concatenate them into a numpy array with shape (n_samples, n_PCs)
    # Note: we just get the std rather than cov, since we are interested in the predictive uncertainty
    #       of a given point, not the correlation between different sample points.
    n_samples = parameters.shape[0]
    emulator_central_value = np.zeros((n_samples, emulator_settings.n_pc))
    emulator_cov = np.zeros((n_samples, emulator_settings.n_pc, emulator_settings.n_pc))

    for i, emulator in enumerate(emulators):
        try:
            # Try to get full covariance matrix
            y_central_value, y_cov = emulator.predict(parameters, return_cov=True)
            emulator_central_value[:, i] = y_central_value

            # y_cov should be shape (n_samples, n_samples) for the covariance between different parameter points
            # We want the diagonal elements which give the variance for each parameter point
            if y_cov.ndim == 2 and y_cov.shape[0] == n_samples and y_cov.shape[1] == n_samples:
                # Extract diagonal variance for each sample
                emulator_cov[:, i, i] = np.diag(y_cov)
            else:
                logger.warning(f"Unexpected covariance shape from emulator {i}: {y_cov.shape}")
                emulator_cov[:, i, i] = np.diag(y_cov) if y_cov.ndim == 2 else y_cov

        except (TypeError, ValueError) as e:
            # Fallback to standard deviation approach if return_cov fails
            logger.warning(f"Failed to get covariance from emulator {i}, falling back to std: {e}")
            y_central_value, y_std = emulator.predict(parameters, return_std=True)
            emulator_central_value[:, i] = y_central_value
            emulator_cov[:, i, i] = y_std**2

    assert emulator_cov.shape == (n_samples, emulator_settings.n_pc, emulator_settings.n_pc)

    # Reconstruct the physical space from the PCs, and invert preprocessing.
    # Note we use array broadcasting to calculate over all samples.
    pca: sklearn_decomposition.PCA = results["PCA"]["pca"]
    scaler: sklearn_preprocessing.StandardScaler = results["PCA"]["scaler"]
    emulator_central_value_reconstructed_scaled = emulator_central_value.dot(
        pca.components_[: emulator_settings.n_pc, :]
    )
    emulator_central_value_reconstructed = scaler.inverse_transform(emulator_central_value_reconstructed_scaled)

    # Propagate uncertainty through the linear transformation back to feature space.
    # Note that for a vector f = Ax, the covariance matrix of f is C_f = A C_x A^T.
    #   (see https://en.wikipedia.org/wiki/Propagation_of_uncertainty)
    #   (Note also that even if C_x is diagonal, C_f will not be)
    # In our case, we have Y[i].T = S*Y_PCA[i].T for each point i in parameter space, where
    #    Y[i].T is a column vector of features -- shape (n_features,)
    #    Y_PCA[i].T is a column vector of corresponding PCs -- shape (n_pc,)
    #    S is the transfer matrix described above -- shape (n_features, n_pc)
    # So C_Y[i] = S * C_Y_PCA[i] * S^T.
    # Note: should be equivalent to: https://github.com/jdmulligan/STAT/blob/master/src/emulator.py#L145
    # TODO: one can make this faster with broadcasting/einsum
    # TODO: NOTE-STAT: Compare this more carefully with STAT L286 and on.
    n_features = pca.components_.shape[1]
    S = pca.components_.T[:, : emulator_settings.n_pc]
    emulator_cov_reconstructed_scaled = np.zeros((n_samples, n_features, n_features))
    for i_sample in range(n_samples):
        emulator_cov_reconstructed_scaled[i_sample] = S.dot(emulator_cov[i_sample].dot(S.T))
    assert emulator_cov_reconstructed_scaled.shape == (n_samples, n_features, n_features)

    # Include predictive variance due to truncated PCs.
    # See comments in mcmc.py for further details.
    for i_sample in range(n_samples):
        emulator_cov_reconstructed_scaled[i_sample] += additional_covariance / n_samples

    # Propagate uncertainty: inverse preprocessing
    # We only need to undo the unit variance scaling, since the shift does not affect the covariance matrix.
    # We can do this by computing an outer product (i.e. product of each pairwise scaling),
    #   and multiplying each element of the covariance matrix by this.
    scale_factors = scaler.scale_
    emulator_cov_reconstructed = emulator_cov_reconstructed_scaled * np.outer(scale_factors, scale_factors)

    # Return the stacked matrices:
    #   Central values: (n_samples, n_features)
    #   Covariances: (n_samples, n_features, n_features)
    emulator_predictions = {}
    emulator_predictions["central_value"] = emulator_central_value_reconstructed
    emulator_predictions["cov"] = emulator_cov_reconstructed

    return emulator_predictions


def compute_additional_covariance_contributions(
    emulator_settings: EmulatorSettings, emulator_result: dict[str, Any]
) -> npt.NDArray[np.float64]:
    """Compute additional contributions to the covariance.

    Args:
        emulator_settings: Emulator settings.
        emulator_result: Emulator training results.
    Returns:
        Unexplained covariance
    """
    # In the case of sk-learn, all we need to handle is the unexplained variance due to PCA truncation.
    return compute_emulator_cov_unexplained(emulator_settings=emulator_settings, emulator_result=emulator_result)


def compute_emulator_cov_unexplained(
    emulator_settings: EmulatorSettings, emulator_result: dict[str, Any]
) -> npt.NDArray[np.float64]:
    """Compute the predictive variance due to PC truncation, for a given emulator.

    We can do this by decomposing the original covariance in feature space:
      C_Y = S D^2 S^T
          = S_{<=n_pc} D^2_{<=n_pc} S_{<=n_pc}^T + S_{>n_pc} D^2_{>n_pc} S_{>n_pc}^T
    In general, we want to estimate the covariance as a function of theta.
    We can do this for the first term by estimating it with the emulator covariance constructed above,
      as a function of theta.
    We can't do this with the second term, since we didn't emulate it -- so we estimate it,
      treating it as independent of theta, and add it to the emulator covariance:
        Sigma_unexplained = 1/n_samples * S_{>n_pc} D^2_{>n_pc} S_{>n_pc}^T,
      where we will include the 1/n_samples factor to account for the fact that we are estimating the covariance from a set of samples.
    See eqs 21-22 of https://arxiv.org/pdf/2102.11337.pdf
    TODO: double check this (and compare to https://github.com/jdmulligan/STAT/blob/master/src/emulator.py#L145)

    We will generally pre-compute this once in the MC sampling to save time, although we define this function
    here to allow us to re-compute it as needed if it is not pre-computed (e.g. when plotting).

    NOTE:
        This is fairly generic functionality for PCA, so it could be ported to other PCA methods.

    Args:
        emulator_settings: Emulator settings.
        emulator_result: Emulator training results.
    Returns:
        Unexplained covariance
    """
    # TODO: NOTE-STAT: Compare this more carefully with STAT L145 and on.
    pca: sklearn_decomposition.PCA = emulator_result["PCA"]["pca"]
    S_unexplained = pca.components_.T[:, emulator_settings.n_pc :]
    D_unexplained = np.diag(pca.explained_variance_[emulator_settings.n_pc :])
    emulator_cov_unexplained = S_unexplained.dot(D_unexplained.dot(S_unexplained.T))

    # NOTE-STAT: bayesian-inference does not include a small term for numerical stability
    return emulator_cov_unexplained  # type: ignore[no-any-return] # noqa: RET504


@attrs.define
class SKLearnEmulatorSettings:
    emulator_name: ClassVar[str] = "sk_learn"
    base_settings: emulation_base.BaseEmulatorSettings
    # PCA settings
    n_pc: int
    max_n_components_to_calculate: int | None
    # Kernels
    active_kernels: dict[str, dict[str, Any]]
    # Gaussian Process Regressor
    n_restarts: int
    alpha: float
    # Keep a copy of the settings for good measure
    settings: dict[str, Any]
    # Additional name, for providing
    additional_name: str = attrs.field(default="")

    def __attrs_post_init__(self):
        """
        Post-creation customization of the emulator configuration.
        """
        # Kernel validation
        # Validate that we have exactly one of matern, rbf
        reference_strings = ["matern", "rbf"]
        assert sum([s in self.active_kernels for s in reference_strings]) == 1, (
            "Must provide exactly one of 'matern', 'rbf' kernel"
        )

        # Validation for noise configuration
        if "noise" in self.active_kernels:
            # Check we have the appropriate keys
            assert [k in self.active_kernels["noise"] for k in ["type", "args"]], (
                "Noise configuration must have keys 'type' and 'args'"
            )
            if self.active_kernels["noise"]["type"] == "white":
                # Validate arguments
                # We don't want to do too much since we'll just be reinventing the wheel, but a bit can be helpful.
                assert set(self.active_kernels["noise"]["args"]) == set(["noise_level", "noise_level_bounds"]), (  # noqa: C405
                    "Must provide arguments 'noise_level' and 'noise_level_bounds' for white noise kernel"
                )
            else:
                msg = "Unsupported noise kernel"
                raise ValueError(msg)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SKLearnEmulatorSettings:
        return cls(
            base_settings=emulation_base.BaseEmulatorSettings.from_emulator_settings(config),
            n_pc=config["n_pc"],
            max_n_components_to_calculate=config.get("max_n_components_to_calculate"),
            active_kernels={kernel_type: config["kernels"][kernel_type] for kernel_type in config["kernels"]["active"]},
            n_restarts=config["GPR"]["n_restarts"],
            alpha=config["GPR"]["alpha"],
            settings=config,
        )

    @classmethod
    def from_config_file(cls, config_file: Path | str, emulator_path: list[str]) -> SKLearnEmulatorSettings:
        """Initialize from the configuration file.

        Args:
            config_file: Path to the configuration file.
            emulator_path: Path to the emulator inside of the configuration file. Need to specify
                the entire path!
        Returns:
            Emulator settings object
        """
        with Path(config_file).open() as stream:
            config = yaml.safe_load(stream)

        # We want the config specific to the emulator, so we need to drill down to just that config.
        def get_nested(d: dict[str, Any], keys_to_follow: list[str]) -> dict[str, Any]:
            for k in keys_to_follow:
                try:
                    d = d[k]
                except KeyError as e:
                    msg = f"Could not find {k=} in dict {d}"
                    raise RuntimeError(msg) from e
            return d

        config = get_nested(d=config, keys_to_follow=emulator_path)

        return cls.from_config(config=config)

    @property
    def force_retrain(self) -> bool:
        # For convenience
        return self.base_settings.force_retrain

    # #---------------------------------------------------------------
    # # Constructor
    # #---------------------------------------------------------------
    # def __init__(self, analysis_name='', parameterization='', analysis_config='', config_file='', emulation_name: str | None = None):

    #     self.analysis_name = analysis_name
    #     self.parameterization = parameterization
    #     self.analysis_config = analysis_config
    #     self.config_file = config_file

    #     with Path(self.config_file).open() as stream:
    #         config = yaml.safe_load(stream)

    #     # Observable inputs
    #     self.observable_table_dir = config['observable_table_dir']
    #     self.observable_config_dir = config['observable_config_dir']
    #     self.observables_filename = config["observables_filename"]

    #     ########################
    #     # Emulator configuration
    #     ########################
    #     if emulation_name is None:
    #         emulator_configuration = self.analysis_config["parameters"]["emulators"]
    #     else:
    #         emulator_configuration = self.analysis_config["parameters"]["emulators"][emulation_name]
    #     self.force_retrain = emulator_configuration['force_retrain']
    #     self.n_pc = emulator_configuration['n_pc']
    #     self.max_n_components_to_calculate = emulator_configuration.get("max_n_components_to_calculate", None)

    #     # Kernels
    #     self.active_kernels = {}
    #     for kernel_type in emulator_configuration['kernels']['active']:
    #         self.active_kernels[kernel_type] = emulator_configuration['kernels'][kernel_type]

    #     # Validate that we have exactly one of matern, rbf
    #     reference_strings = ["matern", "rbf"]
    #     assert sum([s in self.active_kernels for s in reference_strings]) == 1, "Must provide exactly one of 'matern', 'rbf' kernel"

    #     # Validation for noise configuration
    #     if 'noise' in self.active_kernels:
    #         # Check we have the appropriate keys
    #         assert [k in self.active_kernels['noise'] for k in ["type", "args"]], "Noise configuration must have keys 'type' and 'args'"
    #         if self.active_kernels['noise']["type"] == "white":
    #             # Validate arguments
    #             # We don't want to do too much since we'll just be reinventing the wheel, but a bit can be helpful.
    #             assert set(self.active_kernels['noise']["args"]) == set(["noise_level", "noise_level_bounds"]), "Must provide arguments 'noise_level' and 'noise_level_bounds' for white noise kernel"
    #         else:
    #             msg = "Unsupported noise kernel"
    #             raise ValueError(msg)

    #     # GPR
    #     self.n_restarts = emulator_configuration["GPR"]['n_restarts']
    #     self.alpha = emulator_configuration["GPR"]["alpha"]

    #     # Observable list
    #     # None implies a convention of accepting all available data
    #     self.observable_filter = None
    #     observable_list_raw = emulator_configuration.get("observable_list", [])
    #     observable_exclude_list = emulator_configuration.get("observable_exclude_list", [])

    #     # Extract observable names from both old and new config formats
    #     include_list = []
    #     for obs_item in observable_list_raw:
    #         if isinstance(obs_item, str):
    #             # Old format: just the observable name
    #             include_list.append(obs_item)
    #         elif isinstance(obs_item, dict) and 'observable' in obs_item:
    #             # New format: extract observable name from dict
    #             obs_name = obs_item['observable']
    #             include_list.append(obs_name)
    #         else:
    #             logger.warning(f"Unrecognized observable format in emulator config: {obs_item}")

    #     if include_list or observable_exclude_list:
    #         self.observable_filter = data_IO.ObservableFilter(
    #             include_list=include_list,  # Now properly extracted as strings
    #             exclude_list=observable_exclude_list,
    #         )

    #     # Output options
    #     self.output_dir = Path(config['output_dir']) / f'{analysis_name}_{parameterization}'
    #     emulation_outputfile_name = 'emulation.pkl'
    #     if emulation_name is not None:
    #         emulation_outputfile_name = f'emulation_{emulation_name}.pkl'
    #     self.emulation_outputfile = Path(self.output_dir) /  emulation_outputfile_name


# Register the config class as backend entry point
EmulatorSettings = SKLearnEmulatorSettings
