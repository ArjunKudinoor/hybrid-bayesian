"""
Preprocess the input data (eg. outliers removal, smoothing, etc)

authors: J.Mulligan, R.Ehlers, J.Zhang
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt
import yaml

from bayesian import common_base, data_IO, outliers_smoothing
from bayesian.outliers_smoothing import (
    FilteringConfig,
    filter_problematic_design_points,
)

logger = logging.getLogger(__name__)


def preprocess(
    preprocessing_config: PreprocessingConfig,
) -> dict[str, Any]:
    # First, smooth predictions
    observables = smooth_statistical_outliers_in_predictions(
        preprocessing_config=preprocessing_config,
    )
    # Find outliers via ad-hoc measures based on physics expectations
    # steer_find_physics_motivated_outliers(
    #    observables=observables,
    #    preprocessing_config=preprocessing_config,
    # )

    return observables  # noqa: RET504


def steer_find_physics_motivated_outliers(
    observables: dict[str, dict[str, dict[str, Any]]],
    preprocessing_config: PreprocessingConfig,
) -> None:
    for validation_set in [False, True]:
        _find_physics_motivated_outliers(
            observables=observables,
            preprocessing_config=preprocessing_config,
            validation_set=validation_set,
        )


def _find_physics_motivated_outliers(
    observables: dict[str, dict[str, dict[str, Any]]],
    preprocessing_config: PreprocessingConfig,  # noqa: ARG001
    validation_set: bool,
) -> None:
    # Setup
    prediction_key = "Prediction"
    if validation_set:
        prediction_key += "_validation"

    i_design_point_to_exclude: set[int] = set()
    for observable_key in data_IO.sorted_observable_list_from_dict(
        observables[prediction_key],
    ):
        # Get the individual keys from the observable_label
        x = data_IO.observable_label_to_keys(observable_key)  # type: ignore[no-untyped-call]

        # Find all RAAs, and require no points less than 0, and points above 1.3
        if x[2] in ["hadron", "inclusive_chjet", "inclusive_jet"] and (
            not any(subtype in x[3] for subtype in ["Dz", "tg", "zg"])
        ):
            logger.info(f"{observable_key=}")
            i_design_point = np.where(observables[prediction_key][observable_key]["y"] < -0.2)[1]
            logger.info(f"first: {i_design_point=}")
            i_design_point = np.concatenate(
                [i_design_point, np.where(observables[prediction_key][observable_key]["y"] > 1.3)[1]]
            )
            i_design_point_to_exclude.update(i_design_point)

        # What's going on with the theta_g?
        if "tg" in x[3]:
            logger.info(f"{observable_key=}")
            res = np.where(observables[prediction_key][observable_key]["y"] < 0.0)
            logger.info(f"{res=}")
            logger.info(observables[prediction_key][observable_key]["y"][:, res[1]])

    logger.debug(f"{i_design_point_to_exclude=}")
    # TODO: Probably should return the values rather than just print them...
    logger.warning(f"ad-hoc points to exclude: {sorted(i_design_point_to_exclude)}")


def smooth_statistical_outliers_in_predictions(
    preprocessing_config: PreprocessingConfig,
) -> dict[str, Any]:
    """Steer smoothing of statistical outliers in predictions."""
    # Setup for observables
    all_observables = data_IO.read_dict_from_h5(preprocessing_config.output_dir, "observables.h5")

    # Stage 1: Filter design points
    logger.info("Filtering outliers in predictions...")
    filtering_config_dict = preprocessing_config.analysis_config["parameters"]["preprocessing"].get("filtering", {})
    if filtering_config_dict.get("enable", False):
        filtering_config = FilteringConfig(
            method=filtering_config_dict.get("method", "relative_statistical_error"),
            threshold=filtering_config_dict.get("threshold", 0.7),
            min_design_points=filtering_config_dict.get("min_design_points", 50),
            max_filtered_fraction=filtering_config_dict.get("max_filtered_fraction", 0.2),
            problem_fraction_threshold=filtering_config_dict.get("problem_fraction_threshold", 0.3),
        )

        # Filter training set
        logger.info("Filtering training set (Prediction)...")
        all_observables, _filtered_train = filter_problematic_design_points(
            all_observables,
            filtering_config,
            prediction_key="Prediction",  # ← Training set
        )

        # Filter validation set SEPARATELY
        if "Prediction_validation" in all_observables:
            logger.info("Filtering validation set (Prediction_validation)...")
            all_observables, _filtered_val = filter_problematic_design_points(
                all_observables,
                filtering_config,
                prediction_key="Prediction_validation",  # ← Validation set
            )

        logger.info("✓ Filtering stage complete")
    else:
        logger.info("⊗ Filtering disabled")

    # Stage 2: Smoothing design points
    logger.info("Smoothing outliers in predictions...")
    smoothing_config_dict = preprocessing_config.analysis_config["parameters"]["preprocessing"].get("smoothing", {})
    if smoothing_config_dict.get("enable", True):  # Default: True for backward compatibility
        # Continue with existing smoothing code
        new_observables = {}

        new_observables.update(
            _smooth_statistical_outliers_in_predictions(
                all_observables=all_observables,
                validation_set=False,
                preprocessing_config=preprocessing_config,
                outlier_identification_method="large_statistical_errors",
            )
        )

        new_observables.update(
            _smooth_statistical_outliers_in_predictions(
                all_observables=all_observables,
                validation_set=True,
                preprocessing_config=preprocessing_config,
                outlier_identification_method="large_statistical_errors",
            )
        )

        # Merge for large central value differences
        for k in all_observables:
            if k not in new_observables:
                new_observables[k] = all_observables[k]

        new_observables.update(
            _smooth_statistical_outliers_in_predictions(
                all_observables=new_observables,
                validation_set=False,
                preprocessing_config=preprocessing_config,
                outlier_identification_method="large_central_value_difference",
            )
        )

        new_observables.update(
            _smooth_statistical_outliers_in_predictions(
                all_observables=new_observables,
                validation_set=True,
                preprocessing_config=preprocessing_config,
                outlier_identification_method="large_central_value_difference",
            )
        )

        logger.info("✓ Smoothing stage complete")
        return new_observables

    logger.info("⊗ Smoothing disabled (enable: false)")
    return all_observables


def _smooth_statistical_outliers_in_predictions(  # noqa: C901
    all_observables: dict[str, dict[str, dict[str, Any]]],
    validation_set: bool,
    preprocessing_config: PreprocessingConfig,
    outlier_identification_method: str,
) -> dict[str, Any]:
    """Smooth out statistical outliers in predictions.

    Args:
        all_observables: Dictionary of observables from read_dict_from_h5.
        validation_set: Whether to use the validation set or not.
        preprocessing_config: Configuration for preprocessing.
        outlier_identification_mode: How to identify outliers. Options:
            ["large_statistical_error", "large_central_value_difference"]
    """
    # Setup
    prediction_key = "Prediction"
    if validation_set:
        prediction_key += "_validation"

    # These will contain our interpolated predictions
    new_observables: dict[str, dict[str, dict[str, Any]]] = {prediction_key: {}}
    # Outliers which we are unable to remove. We keep track of this to help guide if we need to exclude a design point
    # Format: {observable_key: {design_point: set[feature_index]}
    outliers_we_are_unable_to_remove: dict[str, dict[int, set[int]]] = {}
    for observable_key in data_IO.sorted_observable_list_from_dict(
        all_observables[prediction_key],
    ):
        # First, find the outliers based on the selected method
        if outlier_identification_method == "large_statistical_errors":
            # large statistical uncertainty points
            outliers = outliers_smoothing.find_large_statistical_uncertainty_points(
                values=all_observables[prediction_key][observable_key]["y"],
                y_err=all_observables[prediction_key][observable_key]["y_err_stat"],
                outliers_config=preprocessing_config.smoothing_outliers_config,
            )
        elif outlier_identification_method == "large_central_value_difference":
            # Find additional outliers based on central values which are dramatically different than the others
            if len(all_observables[prediction_key][observable_key]["y"]) > 2:
                outliers = outliers_smoothing.find_outliers_based_on_central_values(
                    values=all_observables[prediction_key][observable_key]["y"],
                    outliers_config=preprocessing_config.smoothing_outliers_config,
                )
            else:
                outliers = ((), ())  # type: ignore[assignment]
        else:
            msg = f"Unrecognized outlier identification mode {outlier_identification_method}."
            raise ValueError(msg)

        # And merge the two together
        # outliers = [  # type: ignore[assignment]
        #    np.concatenate([first, second])
        #    for first, second in zip(outliers, additional_outliers)
        # ]

        # Perform quality assurance and reformat outliers
        outlier_features_to_interpolate_per_design_point, _intermediate_outliers_we_are_unable_to_remove = (
            outliers_smoothing.perform_QA_and_reformat_outliers(
                observable_key=observable_key,
                outliers=outliers,
                smoothing_max_n_feature_outliers_to_interpolate=preprocessing_config.smoothing_max_n_feature_outliers_to_interpolate,
            )
        )
        # Only fill if we actually have something to report
        if observable_key in _intermediate_outliers_we_are_unable_to_remove:
            if observable_key not in outliers_we_are_unable_to_remove:
                outliers_we_are_unable_to_remove[observable_key] = {}
            outliers_we_are_unable_to_remove[observable_key].update(
                _intermediate_outliers_we_are_unable_to_remove[observable_key]
            )

        # Finally, interpolate at the selected outlier point features to find the value and error
        new_observables[prediction_key][observable_key] = {}
        for key_type in ["y", "y_err_stat"]:
            new_observables[prediction_key][observable_key][key_type] = np.array(
                all_observables[prediction_key][observable_key][key_type],
                copy=True,
            )
            observable_bin_centers = (
                all_observables["Data"][observable_key]["xmin"]
                + (all_observables["Data"][observable_key]["xmax"] - all_observables["Data"][observable_key]["xmin"])
                / 2.0
            )
            if len(observable_bin_centers) == 1:
                # Skip - we can't interpolate one point.
                logger.debug(f'Skipping observable "{observable_key}" because it has only one point.')
                continue

            # logger.info(f"Method: {outlier_identification_method}, Interpolating outliers with {outlier_features_to_interpolate_per_design_point=}, {key_type=}, {observable_key=}, {prediction_key=}")
            for design_point, points_to_interpolate in outlier_features_to_interpolate_per_design_point.items():
                try:
                    interpolated_values = outliers_smoothing.perform_interpolation_on_values(
                        bin_centers=observable_bin_centers,
                        values_to_interpolate=new_observables[prediction_key][observable_key][key_type][
                            :, design_point
                        ],
                        points_to_interpolate=points_to_interpolate,
                        smoothing_interpolation_method=preprocessing_config.smoothing_interpolation_method,
                    )
                    new_observables[prediction_key][observable_key][key_type][points_to_interpolate, design_point] = (
                        interpolated_values
                    )
                except outliers_smoothing.CannotInterpolateDueToOnePointError as e:
                    msg = f'Skipping observable "{observable_key}", {design_point=} because {e}'
                    logger.info(msg)
                    # And add to the list since we can't make it work.
                    if observable_key not in outliers_we_are_unable_to_remove:
                        outliers_we_are_unable_to_remove[observable_key] = {}
                    if design_point not in outliers_we_are_unable_to_remove[observable_key]:
                        outliers_we_are_unable_to_remove[observable_key][design_point] = set()
                    outliers_we_are_unable_to_remove[observable_key][design_point].update(points_to_interpolate)
                    continue

    # Reformat the outliers_we_are_unable_to_remove to be more useful and readable
    # logger.info(
    #    f"Observables which we are unable to remove outliers from: {outliers_we_are_unable_to_remove}"
    # )
    # NOTE: The typing is wrong because I based the type annotations on the "Predictions" key only,
    #       since it's more useful here.
    # NOTE: We need to map the i_design_point to the actual design point indices for them to be useful!
    design_point_array: npt.NDArray[np.int64] = all_observables[
        "Design_indices" + ("_validation" if validation_set else "")
    ]  # type: ignore[assignment]
    design_points_we_may_want_to_remove: dict[int, dict[str, set[int]]] = {}
    for observable_key, _v in outliers_we_are_unable_to_remove.items():
        for i_design_point, i_feature in _v.items():
            actual_design_point = design_point_array[i_design_point]
            if actual_design_point not in design_points_we_may_want_to_remove:
                design_points_we_may_want_to_remove[actual_design_point] = {}
            if observable_key not in design_points_we_may_want_to_remove[actual_design_point]:
                design_points_we_may_want_to_remove[actual_design_point][observable_key] = set()
            design_points_we_may_want_to_remove[actual_design_point][observable_key].update(i_feature)
    logger.warning(
        f"Method: {outlier_identification_method}, Design points which we may want to remove: {sorted(design_points_we_may_want_to_remove.keys())}, length: {len(design_points_we_may_want_to_remove)}"
    )
    logger.info(f"In further detail: {design_points_we_may_want_to_remove}")
    return new_observables


@attrs.define
class PreprocessingConfig(common_base.CommonBase):
    analysis_name: str
    parameterization: str
    config_file: Path = attrs.field(converter=Path)
    analysis_config: dict[str, Any] = attrs.field(factory=dict)
    config: dict[str, Any] = attrs.field(init=False)
    output_dir: Path = attrs.field(init=False)

    def __attrs_post_init__(self):
        with self.config_file.open() as stream:
            self.config = yaml.safe_load(stream)

        # Retrieve parameters from the config
        # Smoothing parameters
        smoothing_parameters = self.analysis_config["parameters"]["preprocessing"]["smoothing"]
        self.smoothing_outliers_config = outliers_smoothing.OutliersConfig(n_RMS=smoothing_parameters["outlier_n_RMS"])
        self.smoothing_interpolation_method = smoothing_parameters["interpolation_method"]
        # Validation
        if self.smoothing_interpolation_method not in outliers_smoothing.IMPLEMENTED_INTERPOLATION_METHODS:
            msg = f"Unrecognized interpolation method {self.smoothing_interpolation_method}."
            raise ValueError(msg)
        self.smoothing_max_n_feature_outliers_to_interpolate = smoothing_parameters[
            "max_n_feature_outliers_to_interpolate"
        ]

        # I/O
        self.output_dir = Path(self.config["output_dir"]) / f"{self.analysis_name}_{self.parameterization}"
