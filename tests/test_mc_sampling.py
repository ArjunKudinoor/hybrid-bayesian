"""Tests for MCMC QA diagnostics and sampler settings."""

import numpy as np

from bayesian import mc_sampling as mcmc
from bayesian.mc_sampling import emcee as emcee_sampler


def test_compute_chain_diagnostics_flags_boundary_and_repeats() -> None:
    chain = np.array(
        [
            [[0.10, 0.20], [0.50, 0.50]],
            [[0.20, 0.30], [0.50, 0.50]],
            [[0.30, 0.40], [0.996, 0.50]],
            [[0.40, 0.50], [0.998, 0.50]],
            [[0.50, 0.60], [0.998, 0.50]],
        ],
        dtype=np.float64,
    )

    diagnostics = mcmc.compute_chain_diagnostics(
        chain,
        parameter_min=np.array([0.0, 0.0]),
        parameter_max=np.array([1.0, 1.0]),
        boundary_fraction=0.005,
        late_fraction=0.4,
    )

    near_upper = diagnostics["near_upper_counts"]
    late_near_upper = diagnostics["late_near_upper_counts"]
    longest_repeat = diagnostics["longest_repeat_run"]

    assert near_upper.shape == (2, 2)
    assert int(near_upper[1, 0]) == 3
    assert int(late_near_upper[1, 0]) == 2
    assert int(longest_repeat[1]) == 2
    assert int(longest_repeat[0]) == 0


def test_compute_chain_diagnostics_validates_inputs() -> None:
    chain = np.zeros((4, 2, 2), dtype=np.float64)

    try:
        mcmc.compute_chain_diagnostics(chain, np.array([0.0]), np.array([1.0]))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched parameter bound shapes")

    try:
        mcmc.compute_chain_diagnostics(chain, np.array([0.0, 0.0]), np.array([1.0, 1.0]), boundary_fraction=1.5)
    except ValueError as exc:
        assert "boundary_fraction" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid boundary_fraction")

    try:
        mcmc.compute_chain_diagnostics(chain, np.array([0.0, 0.0]), np.array([1.0, 1.0]), late_fraction=-0.1)
    except ValueError as exc:
        assert "late_fraction" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid late_fraction")


def test_emcee_sampler_settings_reads_random_seed() -> None:
    settings = emcee_sampler.SamplerSettings.from_config(
        {
            "n_walkers": 10,
            "n_burn_steps": 20,
            "n_sampling_steps": 30,
            "n_logging_steps": 5,
            "random_seed": 12345,
        }
    )

    assert settings.random_seed == 12345


def test_emcee_seed_reproduces_short_chain() -> None:
    def log_prob(x):
        x = np.asarray(x)
        return -0.5 * np.sum(x**2)

    seed = 12345
    initial = np.array(
        [
            [0.10, -0.10],
            [0.15, -0.05],
            [-0.10, 0.10],
            [-0.15, 0.05],
        ],
        dtype=np.float64,
    )

    sampler_a = emcee_sampler.LoggingEnsembleSampler(4, 2, log_prob)
    sampler_b = emcee_sampler.LoggingEnsembleSampler(4, 2, log_prob)
    emcee_sampler._apply_random_seed(sampler_a, seed)
    emcee_sampler._apply_random_seed(sampler_b, seed)

    sampler_a.run_mcmc(initial, 8, n_logging_steps=1000)
    sampler_b.run_mcmc(initial, 8, n_logging_steps=1000)

    np.testing.assert_allclose(sampler_a.get_chain(), sampler_b.get_chain())
