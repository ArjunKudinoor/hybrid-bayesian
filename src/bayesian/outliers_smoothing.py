"""Functionality for identifying outliers and smoothing them.

DESIGN POINT FILTERING (Phase 1):
===================================
Filter out problematic design points (entire rows) based on:
- Relative statistical errors across observables
- Identified via FilteringConfig and filter_problematic_design_points()
- Removes design points showing statistical instabilities
- Applies to both training and validation sets separately

OUTLIER SMOOTHING (Phase 2):
==============================
Identify and smooth outliers in individual bins (specific entries):
- Large statistical uncertainty outliers
- Large central value deviation outliers
- Uses interpolation (linear or cubic spline) to smooth
- Applied after filtering, on a per-observable basis

KEY CLASSES:
- OutliersConfig: Configuration for Phase 2 (outlier identification thresholds)
- FilteringConfig: Configuration for Phase 1 (design point filtering)

KEY FUNCTIONS:
Phase 1 (Design Point Filtering):
- filter_problematic_design_points(): Main entry point for design point filtering
- apply_design_point_filtering(): Apply filtering to prediction matrix
- identify_problematic_design_points(): Identify which design points to remove

Phase 2 (Outlier Smoothing):
- find_and_smooth_outliers_standalone(): Main entry point for outlier smoothing
- find_large_statistical_uncertainty_points(): Identify bins with large uncertainties
- find_outliers_based_on_central_values(): Identify bins with unusual values
- smooth_outliers(): Interpolate to smooth identified outliers

DESIGN PHILOSOPHY:
- Phase 1 is conservative: only removes severely problematic design points
- Phase 2 is targeted: only smooths individual problematic bins
- Both phases preserve physics validity while improving emulator training

USAGE:
    # Phase 1: Filter design points
    filtered_observables, filtered_points = filter_problematic_design_points(
        observables, filtering_config, prediction_key='Prediction'
    )

    # Phase 2: Smooth remaining outliers
    smoothed_values, smoothed_errors, removed_outliers = find_and_smooth_outliers_standalone(
        observable_key, bin_centers, values, y_err, outliers_config
    )

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
.. codeauthor:: Jingyu Zhang <jingyu.zhang@cern.ch>, Vanderbilt
"""

from __future__ import annotations

import logging
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt
import scipy  # type: ignore[import]

logger = logging.getLogger(__name__)


IMPLEMENTED_INTERPOLATION_METHODS = ["linear", "cubic_spline"]


@attrs.frozen
class OutliersConfig:
    """Configuration for identifying outliers.

    :param float n_RMS: Number of RMS away from the value to identify as an outlier. Default: 2.
    """

    n_RMS: float = 2.0


def find_large_statistical_uncertainty_points(
    values: npt.NDArray[np.float64],
    y_err: npt.NDArray[np.float64],
    outliers_config: OutliersConfig,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Find problematic points based on large statistical uncertainty points.

    Best to do this observable-by-observable because the relative uncertainty will vary for each one.

    Args:
        values: The values of the observable, for all design points.
        y_err: The uncertainties on the values of the observable, for all design points.
        outliers_config: Configuration for identifying outliers.

    Returns:
        (n_feature_index, n_design_point_index) of identified outliers
    """
    relative_error = y_err / values
    # This is the rms averaged over all of the design points
    rms = np.sqrt(np.mean(relative_error**2, axis=-1))
    # NOTE: Recall that np.where returns (n_feature_index, n_design_point_index) as separate arrays
    outliers = np.where(relative_error > outliers_config.n_RMS * rms[:, np.newaxis])
    return outliers  # type: ignore[return-value] # noqa: RET504


def find_outliers_based_on_central_values(
    values: npt.NDArray[np.float64],
    outliers_config: OutliersConfig,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Find outlier points based on large deviations from close central values."""
    # NOTE: We need abs because we don't care about the sign - we just want a measure.
    diff_between_features = np.abs(np.diff(values, axis=0))
    rms = np.sqrt(np.mean(diff_between_features**2, axis=-1))
    outliers_in_diff_mask = diff_between_features > (outliers_config.n_RMS * rms[:, np.newaxis])
    """
    Now, we need to associate the outliers with the original feature index (ie. taking the diff reduces by one)

    The scheme we'll use to identify problematic points is to take an AND of the left and right of the point.
    For the first and last index, we cannot take an and since they're one sided. To address this point, we'll
    redo the exercise, but with the 1th and -2th removed, and take an AND of those and the original. It's ad-hoc,
    but it gives a second level of cross check for those points.
    """
    # First, we'll handle the inner points
    output = np.zeros_like(values, dtype=np.bool_)
    output[1:-1, :] = outliers_in_diff_mask[:-1, :] & outliers_in_diff_mask[1:, :]

    # Convenient breakpoint for debugging of high values
    # if np.any(values > 1.05):
    #    logger.info(f"{values=}")

    # Now, handle the edges. Here, we need to select the 1th and -2th points
    if values.shape[0] > 4:
        s = np.ones(values.shape[0], dtype=np.bool_)
        s[1] = False
        s[-2] = False
        # Now, we'll repeat the calculation with the diff and rMS
        diff_between_features_for_edges = np.abs(np.diff(values[s, :], axis=0))
        rms = np.sqrt(np.mean(diff_between_features_for_edges**2, axis=-1))
        outliers_in_diff_mask_edges = diff_between_features_for_edges > (outliers_config.n_RMS * rms[:, np.newaxis])
        output[0, :] = outliers_in_diff_mask_edges[0, :] & outliers_in_diff_mask[0, :]
        output[-1, :] = outliers_in_diff_mask_edges[-1, :] & outliers_in_diff_mask[-1, :]
    else:
        # Too short - just have to take what we have
        output[0, :] = outliers_in_diff_mask[0, :]
        output[-1, :] = outliers_in_diff_mask[-1, :]

    # NOTE: Recall that np.where returns (n_feature_index, n_design_point_index) as separate arrays
    outliers = np.where(output)
    return outliers  # type: ignore[return-value] # noqa: RET504


def perform_QA_and_reformat_outliers(
    observable_key: str,
    outliers: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
    smoothing_max_n_feature_outliers_to_interpolate: int,
) -> tuple[dict[int, list[int]], dict[str, dict[int, set[int]]]]:
    """Perform QA on identifier outliers, and reformat them for next steps.

    :param observable_key: The key for the observable we're looking at.
    :param outliers: The outliers provided by the outlier finder.
    :param smoothing_max_n_feature_outliers_to_interpolate: The maximum number of points to interpolate in a row.
    """
    # NOTE: This could skip the observable key, but it's convenient because we then have the same
    #       format as the overall dict
    outliers_we_are_unable_to_remove: dict[str, dict[int, set[int]]] = {}
    # Next, we want to do quality checks.
    # If there are multiple problematic points in a row, we want to skip interpolation since
    # it's not clear that we can reliably interpolate.
    # First, we need to put the features into a more useful order:
    # outliers: zip(feature_index, design_point) -> dict: (design_point, feature_index)
    # NOTE: The `design_point` here is the index in the design point array of the design points
    #       that we've using for this analysis. To actually use them (ie. in print outs), we'll
    #       need to apply them to the actual design point array.
    outlier_features_per_design_point: dict[int, set[int]] = {v: set() for v in outliers[1]}
    for i_feature, design_point in zip(*outliers, strict=True):
        outlier_features_per_design_point[design_point].update([i_feature])
    # These features must be sorted to finding distances between them, but sets are unordered,
    # so we need to explicitly sort them
    for design_point, v in outlier_features_per_design_point.items():
        outlier_features_per_design_point[design_point] = sorted(v)  # type: ignore[assignment]

    # Since the feature values of one design point shouldn't impact another, we'll want to
    # check one design point at a time.
    # NOTE: If we have to skip, we record the design point so we can consider excluding it due
    #       to that observable.
    outlier_features_to_interpolate_per_design_point: dict[int, list[int]] = {}
    # logger.info(f"{observable_key=}, {outlier_features_per_design_point=}")
    for k, v in outlier_features_per_design_point.items():
        # logger.debug("------------------------")
        # logger.debug(f"{k=}, {v=}")
        # Calculate the distance between the outlier indices
        distance_between_outliers = np.diff(list(v))
        # And we'll keep track of which ones pass our quality requirements (not too many in a row).
        indices_of_outliers_that_are_one_apart = set()
        accumulated_indices_to_remove = set()

        for distance, lower_feature_index, upper_feature_index in zip(
            distance_between_outliers, list(v)[:-1], list(v)[1:], strict=True
        ):
            # We're only worried about points which are right next to each other
            if distance == 1:
                indices_of_outliers_that_are_one_apart.update([lower_feature_index, upper_feature_index])
            else:
                # In this case, we now have points that aren't right next to each other.
                # Here, we need to figure out what we're going to do with the points that we've found
                # that **are** right next to each other. Namely, we'll want to remove them from the list
                # to be interpolated, but if there are more points than our threshold.
                # NOTE: We want strictly greater than because we add two points per distance being greater than 1.
                #       eg. one distance(s) of 1 -> two points
                #           two distance(s) of 1 -> three points (due to set)
                #           three distance(s) of 1 -> four points (due to set)
                if len(indices_of_outliers_that_are_one_apart) > smoothing_max_n_feature_outliers_to_interpolate:
                    # Since we are looking at the distances, we want to remove the points that make up that distance.
                    accumulated_indices_to_remove.update(indices_of_outliers_that_are_one_apart)
                else:
                    # For debugging, keep track of when we find points that are right next to each other but
                    # where we skip removing them (ie. keep them for interpolation) because they're below our
                    # max threshold of consecutive points
                    # NOTE: There's no point in warning if empty, since that case is trivial
                    if len(indices_of_outliers_that_are_one_apart) > 0:
                        msg = (
                            f"Will continue with interpolating consecutive indices {indices_of_outliers_that_are_one_apart}"
                            f" because the their number is within the allowable range (n_consecutive<={smoothing_max_n_feature_outliers_to_interpolate})."
                        )
                        logger.info(msg)
                # Reset for the next point
                indices_of_outliers_that_are_one_apart = set()
        # There are indices left over at the end of the loop which we need to take care of.
        # eg. If all points are considered outliers
        if (
            indices_of_outliers_that_are_one_apart
            and len(indices_of_outliers_that_are_one_apart) > smoothing_max_n_feature_outliers_to_interpolate
        ):
            # Since we are looking at the distances, we want to remove the points that make up that distance.
            # logger.info(f"Ended on {indices_of_outliers_that_are_one_apart=}")
            accumulated_indices_to_remove.update(indices_of_outliers_that_are_one_apart)

        # Now that we've determine which points we want to remove from our interpolation (accumulated_indices_to_remove),
        # let's actually remove them from our list.
        # NOTE: We sort again because sets are not ordered.
        outlier_features_to_interpolate_per_design_point[k] = sorted(set(v) - accumulated_indices_to_remove)
        # logger.debug(f"design point {k}: features kept for interpolation: {outlier_features_to_interpolate_per_design_point[k]}")

        # And we'll keep track of what we can't interpolate
        if accumulated_indices_to_remove:
            if observable_key not in outliers_we_are_unable_to_remove:
                outliers_we_are_unable_to_remove[observable_key] = {}
            outliers_we_are_unable_to_remove[observable_key][k] = accumulated_indices_to_remove

    return outlier_features_to_interpolate_per_design_point, outliers_we_are_unable_to_remove


def find_and_smooth_outliers_standalone(
    observable_key: str,
    bin_centers: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    y_err: npt.NDArray[np.float64],
    outliers_identification_methods: dict[str, OutliersConfig],
    smoothing_interpolation_method: str,
    max_n_points_to_interpolate: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[int, set[int]]]:
    """A standalone function to identify outliers and smooth them.

    Careful: If you remove design points, you'll need to make sure to keep careful track of the indices!

    Note:
        For the outliers that we are unable to remove, it's probably best to exclude the design point entirely.
        However, you'll have to take care of it separately.

    Args:
        observable_key: The key for the observable we're looking at. Just a name for bookkeeping.
        bin_centers: The bin centers for the observable.
        values: The values of the observable, for all design points.
        y_err: The uncertainties on the values of the observable, for all design points.
        outliers_identification_methods: The methods to use for identifying outliers. Keys are the methods, while the values
            are the parameters. Key options: {"large_statistical_errors": OutliersConfig, "large_central_value_difference": OutliersConfig}.
        smoothing_interpolation_method: The method to use for interpolation. Options: ["linear", "cubic_spline"].
        max_n_points_to_interpolate: The maximum number of points to interpolate in a row.

    Returns:
        The smoothed values and uncertainties, and the outliers which we are unable to remove ({feature_index: set(design_point_index)}).
    """
    # Validation
    for outlier_identification_method in outliers_identification_methods:
        if outlier_identification_method not in ["large_statistical_errors", "large_central_value_difference"]:
            msg = f"Unrecognized smoothing method {outlier_identification_method}."
            raise ValueError(msg)
    if len(bin_centers) == 1:
        # Skip - we can't interpolate one point.
        msg = f'Skipping observable "{observable_key}" because it has only one point.'
        logger.debug(msg)
        raise ValueError(msg)

    # Setup
    outliers_we_are_unable_to_remove: dict[int, set[int]] = {}
    values = np.array(values, copy=True)
    y_err = np.array(y_err, copy=True)

    # Identify outliers
    # outliers = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    outliers = np.zeros((0, 2), dtype=np.int64)
    for outlier_identification_method, outliers_config in outliers_identification_methods.items():
        # First, find the outliers based on the selected method
        if outlier_identification_method == "large_statistical_errors":
            # large statistical uncertainty points
            new_outliers = find_large_statistical_uncertainty_points(
                values=values,
                y_err=y_err,
                outliers_config=outliers_config,
            )
        elif outlier_identification_method == "large_central_value_difference":
            # Find additional outliers based on central values which are dramatically different than the others
            if len(values) > 2:
                new_outliers = find_outliers_based_on_central_values(
                    values=values,
                    outliers_config=outliers_config,
                )
            else:
                new_outliers = ((), ())  # type: ignore[assignment]
        else:
            msg = f"Unrecognized outlier identification mode {outlier_identification_method}."
            raise ValueError(msg)
        # Merge the outliers together, taking care to deduplicate outlier values that may be stored in each array
        combined_indices = np.concatenate((outliers, np.column_stack(new_outliers)), axis=0)
        outliers = np.unique(combined_indices, axis=0)

    # If needed, can split outliers back into the two arrays
    # outliers_feature_indices, outliers_design_point_indices = outliers[:, 0], outliers[:, 0]
    outlier_features_to_interpolate_per_design_point, _intermediate_outliers_we_are_unable_to_remove = (
        perform_QA_and_reformat_outliers(
            observable_key=observable_key,
            outliers=(outliers[:, 0], outliers[:, 1]),
            smoothing_max_n_feature_outliers_to_interpolate=max_n_points_to_interpolate,
        )
    )
    # And keep track of them
    outliers_we_are_unable_to_remove.update(_intermediate_outliers_we_are_unable_to_remove.get(observable_key, {}))

    # Perform interpolation
    for v in [values, y_err]:
        # logger.info(f"Method: {outlier_identification_method}, Interpolating outliers with {outlier_features_to_interpolate_per_design_point=}, {key_type=}, {observable_key=}, {prediction_key=}")
        for design_point, points_to_interpolate in outlier_features_to_interpolate_per_design_point.items():
            try:
                interpolated_values = perform_interpolation_on_values(
                    bin_centers=bin_centers,
                    values_to_interpolate=v[:, design_point],
                    points_to_interpolate=points_to_interpolate,
                    smoothing_interpolation_method=smoothing_interpolation_method,
                )
                # And assign the interpolated values
                v[points_to_interpolate, design_point] = interpolated_values
            except CannotInterpolateDueToOnePointError as e:
                msg = f'Skipping observable "{observable_key}", {design_point=} because {e}'
                logger.info(msg)
                # And add to the list since we can't make it work.
                if design_point not in outliers_we_are_unable_to_remove:
                    outliers_we_are_unable_to_remove[design_point] = set()
                outliers_we_are_unable_to_remove[design_point].update(points_to_interpolate)
                continue

    return values, y_err, outliers_we_are_unable_to_remove


class CannotInterpolateDueToOnePointError(Exception):
    """Error raised when we can't interpolate due to only one point."""


def perform_interpolation_on_values(
    bin_centers: npt.NDArray[np.float64],
    values_to_interpolate: npt.NDArray[np.float64],
    points_to_interpolate: list[int],
    smoothing_interpolation_method: str,
) -> npt.NDArray[np.float64]:
    """Perform interpolation on the requested points to interpolate.

    Args:
        bin_centers: The bin centers for the observable.
        values_to_interpolate: The values to interpolate.
        points_to_interpolate: The points (i.e. bin centers) to interpolate.
        smoothing_interpolation_method: The method to use for interpolation. Options:
            ["linear", "cubic_spline"].

    Returns:
        The values that are interpolated at points_to_interpolate. They can be inserted into the
            original values_to_interpolate array via `values_to_interpolate[points_to_interpolate] = interpolated_values`.

    Raises:
        CannotInterpolateDueToOnePointError: Raised when we can't interpolate due to only
            one point being left.
    """
    # Validation for methods
    if smoothing_interpolation_method not in IMPLEMENTED_INTERPOLATION_METHODS:
        msg = f"Unrecognized interpolation method {smoothing_interpolation_method}."
        raise ValueError(msg)

    # We want to train the interpolation only on all good points, so we take them out.
    # Otherwise, it will negatively impact the interpolation.
    mask = np.ones_like(bin_centers, dtype=bool)
    mask[points_to_interpolate] = False

    # Further validation
    if len(bin_centers[mask]) == 1:
        # Skip - we can't interpolate one point.
        msg = f"Can't interpolate due to only one point left to anchor the interpolation. {mask=}"
        raise CannotInterpolateDueToOnePointError(msg)

    # NOTE: ROOT::Interpolator uses a Cubic Spline, so this might be a reasonable future approach
    #       However, I think it's slower, so we'll start with this simple approach.
    # TODO: We entirely ignore the interpolation error here. Some approaches for trying to account for it:
    #       - Attempt to combine the interpolation error with the statistical error
    #       - Randomly remove a few percent of the points which are used for estimating the interpolation,
    #         and then see if there are significant changes in the interpolated parameters
    #       - Could vary some parameters (perhaps following the above) and perform the whole
    #         Bayesian analysis, again looking for how much the determined parameters change.
    if smoothing_interpolation_method == "linear":
        interpolated_values = np.interp(
            bin_centers[points_to_interpolate],
            bin_centers[mask],
            values_to_interpolate[mask],
        )
    elif smoothing_interpolation_method == "cubic_spline":
        cs = scipy.interpolate.CubicSpline(
            bin_centers[mask],
            values_to_interpolate[mask],
        )
        interpolated_values = cs(bin_centers[points_to_interpolate])

    return interpolated_values


@attrs.frozen
class FilteringConfig:
    """Configuration for filtering (permanent removal) of design points.

    This is different from OutliersConfig which smooths/interpolates outliers.
    Filtering removes entire design points when they have too many bad features.
    """

    method: str = "relative_statistical_error"  # 'relative_statistical_error', 'absolute_statistical_error'
    threshold: float = 0.5  # Threshold value
    min_design_points: int = 50  # Safety: minimum design points to keep
    max_filtered_fraction: float = 0.3  # Safety: maximum fraction to filter
    problem_fraction_threshold: float = 0.2  # If >20% of features are bad, filter the design point


def identify_high_uncertainty_points_absolute_threshold(
    values: npt.NDArray[np.float64],
    uncertainties: npt.NDArray[np.float64],
    method: str,
    threshold: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Identify high uncertainty points using absolute thresholds.

    Complements existing RMS-based outlier detection with absolute threshold option.

    Args:
        values: Observable values, shape (n_bins, n_design_points)
        uncertainties: Statistical uncertainties, shape (n_bins, n_design_points)
        method: 'relative_statistical_error' or 'absolute_statistical_error'
        threshold: Absolute threshold value

    Returns:
        (feature_indices, design_point_indices) matching existing function signature

    Note:
        Return format matches existing find_large_statistical_uncertainty_points()
        which returns (n_feature_index, n_design_point_index)
    """
    if method == "relative_statistical_error":
        # Filter where |σ / y| > threshold
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_error = np.abs(uncertainties / values)
            relative_error[~np.isfinite(relative_error)] = 0
        mask = relative_error > threshold

    elif method == "absolute_statistical_error":
        # Filter where |σ| > threshold
        mask = np.abs(uncertainties) > threshold

    else:
        msg = f"Unknown filtering method: {method}"
        raise ValueError(msg)

    # np.where returns (row_indices, col_indices)
    # For shape (n_bins, n_design_points): rows=features, cols=design_points
    feature_indices, design_point_indices = np.where(mask)
    return feature_indices, design_point_indices


def identify_design_points_to_filter(
    observables: dict[str, Any],
    config: FilteringConfig,
    prediction_key: str = "Prediction",
) -> list[int]:
    """
    Identify design points that should be completely removed.

    A design point (row in prediction matrix) is marked for removal if it has
    too many problematic features (columns) across all observables.

    Args:
        observables: Observables dictionary
        config: FilteringConfig with filtering parameters
        prediction_key: 'Prediction' or 'Prediction_validation'

    Returns:
        List of design point indices (row indices) to remove
    """
    # Count problematic features per design point
    design_point_problem_count: dict[int, int] = {}
    total_features_per_design_point: dict[int, int] = {}

    for obs_label, obs_data in observables[prediction_key].items():  # noqa: B007
        values = obs_data["y"]  # shape: (n_bins, n_design_points)
        uncertainties = obs_data["y_err_stat"]  # shape: (n_bins, n_design_points)
        n_bins, _n_design_points = values.shape

        # Identify problematic points
        _feature_indices, design_point_indices = identify_high_uncertainty_points_absolute_threshold(
            values, uncertainties, config.method, config.threshold
        )

        # Count problems per design point
        for dp_idx in design_point_indices:
            design_point_problem_count[dp_idx] = design_point_problem_count.get(dp_idx, 0) + 1
            total_features_per_design_point[dp_idx] = total_features_per_design_point.get(dp_idx, 0) + n_bins

    # Determine which design points to filter
    design_points_to_filter = []
    for dp_idx, count in design_point_problem_count.items():
        problem_fraction = count / total_features_per_design_point[dp_idx]
        if problem_fraction > config.problem_fraction_threshold:
            design_points_to_filter.append(dp_idx)
            logger.debug(
                f"Design point {dp_idx}: {count}/{total_features_per_design_point[dp_idx]} "
                f"({problem_fraction:.1%}) features problematic"
            )

    design_points_to_filter = sorted(design_points_to_filter)

    # Safety checks
    if not design_points_to_filter:
        return []

    n_total = next(iter(observables[prediction_key].values()))["y"].shape[1]
    n_filtered = len(design_points_to_filter)
    filtered_fraction = n_filtered / n_total if n_total > 0 else 0

    if filtered_fraction > config.max_filtered_fraction:
        logger.warning(
            f"Filtering would remove {filtered_fraction:.1%} of design points "
            f"(limit: {config.max_filtered_fraction:.1%}). Filtering disabled for safety."
        )
        return []

    if n_total - n_filtered < config.min_design_points:
        logger.warning(
            f"Filtering would leave only {n_total - n_filtered} design points "
            f"(minimum: {config.min_design_points}). Filtering disabled for safety."
        )
        return []

    logger.info(
        f"Identified {n_filtered}/{n_total} design points for filtering ({filtered_fraction:.1%}): "
        f"{design_points_to_filter}"
    )
    return design_points_to_filter


def apply_design_point_filtering(  # noqa: C901
    observables: dict[str, Any],
    design_points_to_filter: list[int],
    prediction_key: str = "Prediction",
) -> dict[str, Any]:
    """
    Apply design point filtering to observables dictionary.

    This removes columns from the prediction arrays (axis=1).
    Design points are stored as columns in the raw format.

    Args:
        observables: Input observables dictionary
        design_points_to_filter: List of design point indices (column indices) to remove
        prediction_key: 'Prediction' or 'Prediction_validation'

    Returns:
        Filtered observables dictionary
    """
    if not design_points_to_filter:
        return observables

    logger.info(f"Applying filtering to {prediction_key}: removing {len(design_points_to_filter)} design points")

    filtered_observables = {}

    # Copy everything EXCEPT the keys we're explicitly filtering
    keys_to_filter = [prediction_key]
    if prediction_key == "Prediction":
        keys_to_filter.extend(["Design", "Design_indices"])
    elif prediction_key == "Prediction_validation":
        keys_to_filter.extend(["Design_validation", "Design_indices_validation"])

    for key, val in observables.items():
        if key not in keys_to_filter:
            filtered_observables[key] = val

    # Determine which Design/Design_indices keys to filter
    if prediction_key == "Prediction":
        design_key = "Design"
        indices_key = "Design_indices"
    elif prediction_key == "Prediction_validation":
        design_key = "Design_validation"
        indices_key = "Design_indices_validation"
    else:
        msg = f"Unknown prediction_key: {prediction_key}"
        raise ValueError(msg)

    # Filter Prediction arrays
    filtered_observables[prediction_key] = {}
    for obs_label, obs_data in observables[prediction_key].items():
        filtered_obs_data = {}

        n_design_points = obs_data["y"].shape[1]
        keep_mask = np.ones(n_design_points, dtype=bool)
        keep_mask[design_points_to_filter] = False

        for key, value in obs_data.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                # Filter columns (design points)
                filtered_obs_data[key] = value[:, keep_mask]
            elif isinstance(value, dict):
                # Systematics
                filtered_systematics = {}
                for sys_name, sys_value in value.items():
                    if isinstance(sys_value, np.ndarray) and sys_value.ndim == 2:
                        filtered_systematics[sys_name] = sys_value[:, keep_mask]
                    else:
                        filtered_systematics[sys_name] = sys_value
                filtered_obs_data[key] = filtered_systematics  # type: ignore[assignment]
            else:
                filtered_obs_data[key] = value

        filtered_observables[prediction_key][obs_label] = filtered_obs_data

    # Filter corresponding Design array (rows)
    if design_key in observables:
        design = observables[design_key]
        keep_mask = np.ones(design.shape[0], dtype=bool)
        keep_mask[design_points_to_filter] = False
        filtered_observables[design_key] = design[keep_mask, :]

    # Filter corresponding Design_indices
    if indices_key in observables:
        indices = observables[indices_key]
        keep_mask = np.ones(len(indices), dtype=bool)
        keep_mask[design_points_to_filter] = False
        filtered_observables[indices_key] = indices[keep_mask]

    # Copy other Design/indices keys unchanged
    for key in ["Design", "Design_indices", "Design_validation", "Design_indices_validation"]:
        if key in observables and key != design_key and key != indices_key:
            filtered_observables[key] = observables[key]

    logger.info(f"  {prediction_key}: {n_design_points} → {np.sum(keep_mask)} design points")
    if design_key in observables:
        logger.info(f"  {design_key}: {observables[design_key].shape} → {filtered_observables[design_key].shape}")

    return filtered_observables


def filter_problematic_design_points(
    observables: dict[str, Any],
    filtering_config: FilteringConfig,
    prediction_key: str = "Prediction",
) -> tuple[dict[str, Any], list[int]]:
    """
    High-level interface: Filter design points with excessive uncertainty.

    This complements the existing smoothing workflow:
    - Existing: Find outliers → Interpolate → Keep all design points
    - New: Find design points with many outliers → Remove entirely

    Usage: Call this BEFORE smoothing to remove worst design points,
           then smooth remaining mild outliers.

    Args:
        observables: Input observables dictionary
        filtering_config: Configuration for filtering
        prediction_key: 'Prediction' or 'Prediction_validation'

    Returns:
        (filtered_observables, list of removed design point indices)

    Example:
        >>> config = FilteringConfig(
        ...     method='relative_statistical_error',
        ...     threshold=0.7,  # Remove if >70% relative error
        ...     problem_fraction_threshold=0.3  # If >30% of features are bad
        ... )
        >>> filtered_obs, removed = filter_problematic_design_points(obs, config)
        >>> # Then continue with existing smoothing...
        >>> smoothed_obs = find_and_smooth_outliers_standalone(filtered_obs, ...)
    """
    logger.info("=" * 70)
    logger.info("Filtering problematic design points")
    logger.info(f"  Method: {filtering_config.method}")
    logger.info(f"  Threshold: {filtering_config.threshold}")
    logger.info(f"  Problem fraction threshold: {filtering_config.problem_fraction_threshold}")
    logger.info("=" * 70)

    design_points_to_filter = identify_design_points_to_filter(observables, filtering_config, prediction_key)

    if design_points_to_filter:
        filtered_obs = apply_design_point_filtering(observables, design_points_to_filter, prediction_key)
        logger.info(f"✓ Removed {len(design_points_to_filter)} design points")
        logger.info("=" * 70)
        return filtered_obs, design_points_to_filter
    logger.info("✓ No design points need filtering")
    logger.info("=" * 70)
    return observables, []
