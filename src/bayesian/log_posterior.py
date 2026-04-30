"""Define the likelihood separately for performance reasons

In doing so, we can use global variables. This isn't a nice thing to do from a coding perspective,
but it gives a significant improvement in MCMC performance during multiprocessing.
For the initial concept, see: https://emcee.readthedocs.io/en/stable/tutorials/parallel/#parallel

SYSTEMATIC UNCERTAINTY SUPPORT (August - November 2025, Jingyu Zhang)::
================================
Updated to use correlation-aware experimental data structure and systematic covariance matrices.

For detailed information on systematic correlation structure, see systematic_correlation.py
For covariance matrix visualization, see plot_covariance.py

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
.. codeauthor:: James Mulligan
.. codeauthor:: Jingyu Zhang <jingyu.zhang@cern.ch>, Vanderbilt
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy.linalg import lapack

from bayesian import emulation

logger = logging.getLogger(__name__)


g_min: npt.NDArray[np.float64] = None
g_max: npt.NDArray[np.float64] = None
g_emulation_config: emulation.EmulationConfig = None
g_emulation_results: dict[str, dict[str, npt.NDArray[np.float64]]] = None
g_experimental_results: dict = None
g_emulator_cov_unexplained: dict = None


def initialize_pool_variables(
    local_min,
    local_max,
    local_emulation_config,
    local_emulation_results,
    local_experimental_results,
    local_emulator_cov_unexplained,
) -> None:
    """
    Initialize global variables for multiprocessing pool.

    CHANGES August 2025
    - Calculate systematic covariance matrix once during initialization
    - Store in global variable for efficient reuse during MCMC
    """

    global g_min
    global g_max
    global g_emulation_config
    global g_emulation_results
    global g_experimental_results
    global g_emulator_cov_unexplained
    global g_systematic_covariance
    g_min = local_min
    g_max = local_max
    g_emulation_config = local_emulation_config
    g_emulation_results = local_emulation_results
    g_experimental_results = local_experimental_results
    g_emulator_cov_unexplained = local_emulator_cov_unexplained

    # Calculate systematic covariance matrix once during initialization
    # This is efficient because systematic covariance doesn't depend on parameter values

    # Check for external covariance first (expert mode)
    if "external_covariance" in g_experimental_results:
        logger.info("External covariance mode: skipping systematic covariance construction")
        g_systematic_covariance = None  # Not used in external mode

        ext_cov = g_experimental_results["external_covariance"]

        # Check if positive definite
        eigenvals = np.linalg.eigvals(ext_cov)
        logger.info(f"External covariance eigenvalues: min={np.min(eigenvals):.6e}, max={np.max(eigenvals):.6e}")
        if np.min(eigenvals) <= 0:
            logger.error("External covariance is NOT positive definite!")

    elif "correlation_manager" in g_experimental_results:
        logger.info("Calculating systematic covariance matrix for MCMC...")
        correlation_manager = g_experimental_results["correlation_manager"]
        systematic_uncertainties = g_experimental_results["y_err_syst"]
        systematic_names = g_experimental_results["systematic_names"]
        n_features = len(g_experimental_results["y"])

        g_systematic_covariance = correlation_manager.create_systematic_covariance_matrix(
            systematic_uncertainties, systematic_names, n_features
        )

    else:
        logger.info("No correlation manager found - using diagonal systematic uncertainties")
        # Fallback: create diagonal systematic covariance if available
        if "y_err_syst" in g_experimental_results and g_experimental_results["y_err_syst"].shape[1] > 0:
            n_features = len(g_experimental_results["y"])
            g_systematic_covariance = np.zeros((n_features, n_features))
            # Sum over systematic sources (assuming full correlation within each source)
            for sys_idx in range(g_experimental_results["y_err_syst"].shape[1]):
                sys_errors = g_experimental_results["y_err_syst"][:, sys_idx]
                g_systematic_covariance += np.outer(sys_errors, sys_errors)
        else:
            # No systematic uncertainties
            n_features = len(g_experimental_results["y"])
            g_systematic_covariance = np.zeros((n_features, n_features))

    # Build statistical covariance (may include per-observable external stat cov)
    if g_experimental_results.get("per_observable_external_stat_cov"):
        # Build with per-observable external stat cov blocks
        n_features = len(g_experimental_results["y"])
        C_stat = np.zeros((n_features, n_features))

        # Fill diagonal with standard statistical errors by default
        np.fill_diagonal(C_stat, g_experimental_results["y_err_stat"] ** 2)

        # Replace blocks with external stat cov where specified
        for obs_label, cov_info in g_experimental_results["per_observable_external_stat_cov"].items():
            start = cov_info["start"]
            end = cov_info["end"]
            ext_cov = cov_info["matrix"]

            # Replace the block
            C_stat[start:end, start:end] = ext_cov

            logger.info(f"Saved external stat cov for {obs_label} (bins {start}-{end}) to plotting file")

    else:
        # Use diagonal statistical errors
        C_stat = np.diag(g_experimental_results["y_err_stat"] ** 2)

    covariance_matrices = {
        "statistical": C_stat,  # Now includes per-observable external stat cov if present
        "systematic_total": g_systematic_covariance,
        "emulator": None,  # Will be filled with emulator predictions
    }

    # Save to a separate file for plotting
    import pickle
    from pathlib import Path

    output_dir = Path(g_emulation_config.output_dir)
    output_file = output_dir / "covariance_matrices.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(covariance_matrices, f)
    logger.info(f"Saved covariance matrices to {output_file}")


# ---------------------------------------------------------------
def log_posterior(X, *, set_to_infinite_outside_bounds: bool = True) -> npt.NDArray[np.float64]:
    """
    Function to evaluate the log-posterior for a given set of input parameters.

    This function is called by https://emcee.readthedocs.io/en/stable/user/sampler/

    CHANGES August 2025
    - Updated data access to use 'y_err_stat' instead of 'y_err'
    - Added systematic covariance matrix to likelihood calculation
    - Maintains backward compatibility

    :param X input ndarray of parameter space values
    :param min list of minimum boundaries for each emulator parameter
    :param max list of maximum boundaries for each emulator parameter
    :param config emulation_configuration object
    :param emulation_results dict of emulation groups
    :param experimental_results arrays of experimental results
    """

    # Convert to 2darray of shape (n_samples, n_parameters)
    X = np.array(X, copy=False, ndmin=2)

    # Initialize log-posterior array, which we will populate and return
    log_posterior = np.zeros(X.shape[0])

    # Check if any samples are outside the parameter bounds, and set log-posterior to -inf for those
    inside = np.all((X > g_min) & (X < g_max), axis=1)  # noqa: SIM300
    # -1e300 is apparently preferred for pocoMC
    log_posterior[~inside] = -np.inf if set_to_infinite_outside_bounds else -1e300

    # Evaluate log-posterior for samples inside parameter bounds
    n_samples = np.count_nonzero(inside)
    n_features = g_experimental_results["y"].shape[0]

    if n_samples > 0:
        # Get experimental data
        data_y = g_experimental_results["y"]
        data_y_err = g_experimental_results["y_err_stat"]

        # Compute emulator prediction
        # Returns dict of matrices of emulator predictions:
        #     emulator_predictions['central_value'] -- (n_samples, n_features)
        #     emulator_predictions['cov'] -- (n_samples, n_features, n_features)
        emulator_predictions = emulation.predict(
            X[inside],
            g_emulation_config,
            emulator_results=g_emulation_results,
            emulator_additional_covariance=g_emulator_cov_unexplained,
        )

        # Construct array to store the difference between emulator prediction and experimental data
        # (using broadcasting to subtract each data point from each emulator prediction)
        assert data_y.shape[0] == emulator_predictions["central_value"].shape[1]
        dY = emulator_predictions["central_value"] - data_y

        # Construct the covariance matrix
        # NOTE-STAT TODO: include full experimental data covariance matrix -- currently we only include uncorrelated data uncertainty
        # -------------------------
        # Construct the covariance matrix
        covariance_matrix = np.zeros((n_samples, n_features, n_features))
        covariance_matrix += emulator_predictions["cov"]

        # Add experimental uncertainty based on mode
        if "external_covariance" in g_experimental_results:
            # MODE 1: External covariance (expert mode)
            covariance_matrix += g_experimental_results["external_covariance"][np.newaxis, :, :]

            if np.any(~np.isfinite(covariance_matrix)):
                logger.error("Non-finite values in covariance matrix!")
        else:
            # MODE 2 & 3: Standard mode (stat + sys)

            # Check for per-observable external stat covariances
            if g_experimental_results.get("per_observable_external_stat_cov"):
                # Build statistical covariance with per-observable external cov where specified
                C_stat = np.zeros((n_features, n_features))

                # Fill diagonal with standard statistical errors by default
                np.fill_diagonal(C_stat, data_y_err**2)

                # Replace blocks with external stat cov where specified
                for obs_label, cov_info in g_experimental_results["per_observable_external_stat_cov"].items():
                    start = cov_info["start"]
                    end = cov_info["end"]
                    ext_cov = cov_info["matrix"]

                    # Replace the block
                    C_stat[start:end, start:end] = ext_cov

                covariance_matrix += C_stat[np.newaxis, :, :]
            else:
                # Use diagonal statistical errors
                covariance_matrix += np.diag(data_y_err**2)

            # Add systematic covariance matrix (same for all parameter points)
            if g_systematic_covariance is not None:
                covariance_matrix += g_systematic_covariance[np.newaxis, :, :]

        # Compute log likelihood at each point in the sample
        # We take constant priors, so the log-likelihood is just the log-posterior
        # (since above we set the log-posterior to -inf for samples outside the parameter bounds)
        log_posterior[inside] += list(map(_loglikelihood, dY, covariance_matrix))

        # NOTE-STAT: We don't support the extra_std term here.

    return log_posterior


# ---------------------------------------------------------------
def _loglikelihood(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.
    The normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        msg = "lapack dpotrf error: "
        msg += f"the {-info}-th argument had an illegal value"
        raise ValueError(msg)
    if info < 0:
        msg = "lapack dpotrf error: "
        msg += f"the leading minor of order {info} is not positive definite"
        raise np.linalg.LinAlgError(msg)

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        msg = "lapack dpotrs error: "
        msg += f"the {-info}-th argument had an illegal value"
        raise ValueError()

    return -0.5 * np.dot(y, alpha) - np.log(L.diagonal()).sum()
