"""Tests for standalone smoothing functions.

"""

from __future__ import annotations

import logging

import numpy as np
import pytest  # noqa: F401

from bayesian import outliers_smoothing

logger = logging.getLogger(__name__)


def test_smoothing() -> None:
    # Build a small synthetic observable with one obvious outlier.
    bin_centers = np.linspace(20.0, 70.0, 6)
    values = np.tile(np.arange(10.0, 16.0)[:, np.newaxis], (1, 10))
    values = values + np.linspace(0.0, 0.9, 10)
    values[2, 5] = 40.0
    y_err = np.full_like(values, 0.2)
    y_err[2, 5] = 8.0

    # Identify outliers and smooth them
    output_values, output_y_err, outliers_that_cannot_be_removed = outliers_smoothing.find_and_smooth_outliers_standalone(
        observable_key="hadron__pt_ch_cms",
        bin_centers=bin_centers,
        values=values,
        y_err=y_err,
        # Default values as of September 2024
        outliers_identification_methods={
            "large_statistical_errors": outliers_smoothing.OutliersConfig(n_RMS=2),
            "large_central_value_difference": outliers_smoothing.OutliersConfig(n_RMS=2),
        },
        smoothing_interpolation_method="linear",
        max_n_points_to_interpolate=2,
    )

    assert not np.allclose(output_values, values)
    assert not np.allclose(output_y_err, y_err)
    assert output_values.shape == values.shape
    assert output_y_err.shape == y_err.shape
    assert not outliers_that_cannot_be_removed
