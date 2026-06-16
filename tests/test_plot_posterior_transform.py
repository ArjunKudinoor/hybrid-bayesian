"""Tests for declarative posterior transform plotting helpers."""

import numpy as np

from bayesian import plot_posterior_transform


def test_transform_samples_matches_lres_eloss_mapping() -> None:
    samples = np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float64)
    transform_config = {
        "components": [
            {
                "name": "Lres",
                "steps": [
                    {"kind": "multiply", "value": np.pi / 2.0},
                    {"kind": "tan"},
                    {"kind": "power", "value": 3.0},
                ],
            },
            {
                "name": "kappa_sc",
                "steps": [
                    {"kind": "multiply", "value": 0.3},
                    {"kind": "add", "value": 0.3},
                ],
            },
        ]
    }

    transformed = plot_posterior_transform.transform_samples(samples, transform_config)

    expected = np.column_stack(
        [
            np.tan(np.pi / 2.0 * samples[:, 0]) ** 3,
            0.3 + 0.3 * samples[:, 1],
        ]
    )
    np.testing.assert_allclose(transformed, expected)


def test_transform_samples_supports_inverse_steps() -> None:
    samples = np.array([[0.1, 0.2], [0.6, 0.9]], dtype=np.float64)
    transform_config = {
        "components": [
            {
                "name": "a",
                "steps": [{"kind": "multiply", "value": 2.0}, {"kind": "add", "value": 1.0}],
                "inverse_steps": [{"kind": "subtract", "value": 1.0}, {"kind": "divide", "value": 2.0}],
            },
            {
                "name": "b",
                "steps": [{"kind": "multiply", "value": 0.3}, {"kind": "add", "value": 0.3}],
                "inverse_steps": [{"kind": "subtract", "value": 0.3}, {"kind": "divide", "value": 0.3}],
            },
        ]
    }

    physical = plot_posterior_transform.transform_samples(samples, transform_config, direction="forward")
    recovered = plot_posterior_transform.transform_samples(physical, transform_config, direction="inverse")

    np.testing.assert_allclose(recovered, samples)


def test_normalize_random_seed_treats_negative_as_default() -> None:
    assert plot_posterior_transform._normalize_random_seed(None) == 12345
    assert plot_posterior_transform._normalize_random_seed(-1) == 12345
    assert plot_posterior_transform._normalize_random_seed(7) == 7


def test_values_for_log_axis_drop_invalid_entries() -> None:
    values = np.array([0.0, 1.0, np.inf, 2.0, -3.0], dtype=np.float64)
    filtered = plot_posterior_transform._values_for_axis_scale(values, "log", "unit-test")
    np.testing.assert_allclose(filtered, np.array([1.0, 2.0]))


def test_paired_values_for_log_axis_drop_invalid_entries() -> None:
    x = np.array([0.0, 1.0, 2.0, np.inf], dtype=np.float64)
    y = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float64)
    filtered_x, filtered_y = plot_posterior_transform._paired_values_for_axis_scale(
        x, y, "log", "linear", "unit-test"
    )
    np.testing.assert_allclose(filtered_x, np.array([2.0]))
    np.testing.assert_allclose(filtered_y, np.array([3.0]))
