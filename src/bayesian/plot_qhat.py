"""
Module related to generating qhat plots

authors: J.Mulligan, R.Ehlers
"""

import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from bayesian import data_IO, emulation, plot_utils
from bayesian import mc_sampling as mcmc

sns.set_context("paper", rc={"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18})

logger = logging.getLogger(__name__)


####################################################################################################################
def plot(config):
    """
    Generate qhat plots, using data written to mcmc.h5 file in analysis step.
    If no file is found at expected location, no plotting will be done.

    :param MCMCConfig config: we take an instance of MCMCConfig as an argument to keep track of config info.
    """

    # Check if mcmc.h5 file exists
    if not Path(config.mcmc_outputfile).exists():
        logger.info(f"MCMC output does not exist: {config.mcmc_outputfile}")
        return

    # Get results from file
    results = data_IO.read_dict_from_h5(config.output_dir, config.mcmc_outputfilename, verbose=True)
    n_walkers, n_steps, n_params = results["chain"].shape
    posterior = results["chain"].reshape((n_walkers * n_steps, n_params))

    # Plot output dir
    plot_dir = Path(config.output_dir) / "plot_qhat"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # qhat plots
    plot_qhat(posterior, plot_dir, config, E=100, cred_level=0.9, n_samples=1000)
    plot_qhat(posterior, plot_dir, config, T=0.3, cred_level=0.9, n_samples=1000)

    # Observable sensitivity plots
    _plot_observable_sensitivity(posterior, plot_dir, config, delta=0.1, n_samples=1000)


# ---------------------------------------------------------------[]
def plot_qhat(
    posterior,
    plot_dir,
    config,
    E=0,
    T=0,
    cred_level=0.0,
    n_samples=5000,
    n_x=50,
    plot_prior=True,
    plot_mean=True,
    plot_map=False,
    target_design_point: npt.NDArray[np.int64] | None = None,
):
    """
    Plot qhat credible interval from posterior samples,
    as a function of either E or T (with the other held fixed)

    :param 2darray posterior: posterior samples -- shape (n_walkers*n_steps, n_params)
    :param float E: fix jet energy (GeV), and plot as a function of T
    :param float T: fix temperature (GeV), and plot as a function of E
    :param float cred_level: credible interval level
    :param int n_samples: number of posterior samples to use for plotting
    :param int n_x: number of T or E points to plot
    :param 1darray target_design_point: if closure test, design point corresponding to "truth" qhat value
    """
    # Validation
    if target_design_point is None:
        target_design_point = np.array([])

    # Sample posterior parameters without replacement
    if posterior.shape[0] < n_samples:
        n_samples = posterior.shape[0]
        logger.warning(f"Not enough posterior samples to plot {n_samples} samples, using {n_samples} instead")
    rng = np.random.default_rng()
    idx = rng.choice(posterior.shape[0], size=n_samples, replace=False)
    posterior_samples = posterior[idx, :]

    # Compute qhat for each sample (as well as MAP value), as a function of T or E
    #   qhat_posteriors will be a 2d array of shape (x_array.size, n_samples)
    if E:
        xlabel = "T (GeV)"
        suffix = f"E{E}"
        label = f"E = {E} GeV"
        x_array = np.linspace(0.16, 0.5, n_x)
        qhat_posteriors = np.array([qhat_over_T_cubed(posterior_samples, config, T=T, E=E) for T in x_array])
    elif T:
        xlabel = "E (GeV)"
        suffix = f"T{T}"
        label = f"T = {T} GeV"
        x_array = np.linspace(5, 200, n_x)
        qhat_posteriors = np.array([qhat_over_T_cubed(posterior_samples, config, T=T, E=E) for E in x_array])

    # Plot mean qhat values for each T or E
    if plot_mean:
        qhat_mean = np.mean(qhat_posteriors, axis=1)
        plt.plot(x_array, qhat_mean, sns.xkcd_rgb["denim blue"], linewidth=2.0, linestyle="--", label="Mean")

    # Plot the MAP value as well for each T or E
    if plot_map:
        if E:
            qhat_map = np.array(
                [qhat_over_T_cubed(mcmc.map_parameters(posterior_samples), config, T=T, E=E) for T in x_array]
            )
        elif T:
            qhat_map = np.array(
                [qhat_over_T_cubed(mcmc.map_parameters(posterior_samples), config, T=T, E=E) for E in x_array]
            )
        plt.plot(x_array, qhat_map, sns.xkcd_rgb["medium green"], linewidth=2.0, linestyle="--", label="MAP")

    # Get credible interval for each T or E
    h = [mcmc.credible_interval(qhat_values, confidence=cred_level) for qhat_values in qhat_posteriors]
    credible_low = [i[0] for i in h]
    credible_up = [i[1] for i in h]
    plt.fill_between(
        x_array,
        credible_low,
        credible_up,
        color=sns.xkcd_rgb["light blue"],
        label=f"Posterior {int(cred_level * 100)}% Credible Interval",
    )

    # Plot prior as well, for comparison
    # TODO: one could also plot some type of "information gain" metric, e.g. KL divergence
    if plot_prior:
        # Generate samples
        prior_samples = _generate_prior_samples(config, n_samples=n_samples)

        # Compute qhat for each sample, as a function of T or E
        if E:
            qhat_priors = np.array([qhat_over_T_cubed(prior_samples, config, T=T, E=E) for T in x_array])
        elif T:
            qhat_priors = np.array([qhat_over_T_cubed(prior_samples, config, T=T, E=E) for E in x_array])

        # Get credible interval for each T or E
        h_prior = [mcmc.credible_interval(qhat_values, confidence=cred_level) for qhat_values in qhat_priors]
        credible_low_prior = [i[0] for i in h_prior]
        credible_up_prior = [i[1] for i in h_prior]
        plt.fill_between(
            x_array,
            credible_low_prior,
            credible_up_prior,
            color=sns.xkcd_rgb["light blue"],
            alpha=0.3,
            label=f"Prior {int(cred_level * 100)}% Credible Interval",
        )

    # If closure test: Plot truth qhat value
    # We will return a dict of info needed for plotting closure plots, including a
    #   boolean array (as a fcn of T or E) of whether the truth value is contained within credible region
    if target_design_point.any():
        if E:
            qhat_truth = [qhat_over_T_cubed(target_design_point, config, T=T, E=E) for T in x_array]
        elif T:
            qhat_truth = [qhat_over_T_cubed(target_design_point, config, T=T, E=E) for E in x_array]
        plt.plot(x_array, qhat_truth, sns.xkcd_rgb["pale red"], linewidth=2.0, label="Target")

        qhat_closure = {}
        qhat_closure["qhat_closure_array"] = np.array(
            [((qhat_truth[i] < credible_up[i]) and (qhat_truth[i] > credible_low[i])) for i, _ in enumerate(x_array)]
        ).squeeze()
        qhat_closure["qhat_mean"] = qhat_mean
        qhat_closure["x_array"] = x_array
        qhat_closure["cred_level"] = cred_level

    # Plot formatting
    plt.xlabel(xlabel)
    plt.ylabel(r"$\hat{q}/T^3$")
    ymin = 0
    if plot_mean:
        ymax = 2 * max(qhat_mean)
    elif plot_map:
        ymax = 2 * max(qhat_map)
    axes = plt.gca()
    # axes.set_ylim([ymin, ymax])
    axes.set_ylim([0, 12])
    plt.legend(title=f"{label}", title_fontsize=12, loc="upper right", fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/qhat_{suffix}.pdf")
    plt.close("all")

    if target_design_point.any():
        return qhat_closure


# ---------------------------------------------------------------
def _plot_observable_sensitivity(posterior, plot_dir, config, delta=0.1, n_samples=1000):
    """
    Plot local sensitivity index (at the MAP value) for each parameter x_i to each observable O_j:
        S(x_i, O_j, delta) = 1/delta * [O_j(x_i') - O_j(x_i)] / O_j(x_i)
    where x_i'=(1+delta)x_i and delta is a fixed parameter.

    Note: this is just a normalized partial derivative, dO_j/dx_i * (x_i/O_j)

    Based on:
      - https://arxiv.org/abs/2011.01430
      - https://link.springer.com/article/10.1007/BF00547132
    """

    # Find the MAP value
    map_parameters = mcmc.map_parameters(posterior)

    # Plot sensitivity index for each parameter
    for i_parameter in range(posterior.shape[1]):
        _plot_single_parameter_observable_sensitivity(map_parameters, i_parameter, plot_dir, config, delta=delta)

    # TODO: Plot sensitivity for qhat:
    #   S(qhat, O_j, delta) = 1/delta * [O_j(qhat_map') - O_j(qhat_map)] / O_j(qhat)
    # In the current qhat formulation, qhat = qhat(x_0=alpha_s_fix) only depends on x_0=alpha_s_fix.
    # So this information is already captured in the x_0 sensitivity plot above.
    # If we want to explicitly compute S(qhat), we need to evaluate the emulator at qhat_map'=(1+delta)*qhat_map.
    # In principle one should find the x_0 corresponding to (1+delta)*qhat_map.
    # For simplicity we can just evaluate x_0'=x_0(1+delta) and then redefine delta=qhat(x_0')-qhat(x_0) -- but
    #   this is exactly the same as the S(x_0) plot above, up the redefinition of delta.
    # It may nevertheless be nice to add since a plot of S(qhat) will likely be more salient to viewers.


# ---------------------------------------------------------------
def _plot_single_parameter_observable_sensitivity(map_parameters, i_parameter, plot_dir, config, delta=0.1):
    """
    Plot local sensitivity index (at the MAP value) for a single parameter x_i to each observable O_j:
        S(x_i, O_j, delta) = 1/delta * [O_j(x_i') - O_j(x_i)] / O_j(x_i)
    where x_i'=(1+delta)x_i and delta is a fixed parameter.

    TODO: We probably want to add higher level summary plot, e.g. take highest 5 observables for each parameter,
            or average bins over a given observable.
          Could also construct a 2D plot where color shows the sensitivity
    """

    # Define the two parameter points we would like to evaluate
    x = map_parameters.copy()
    x_prime = map_parameters.copy()
    x_prime[i_parameter] = (1 + delta) * x_prime[i_parameter]
    x = np.expand_dims(x, axis=0)
    x_prime = np.expand_dims(x_prime, axis=0)

    # Get emulator predictions at the two points
    emulation_config = emulation.EmulationConfig.from_config_file(
        analysis_settings=config.analysis_settings,
    )
    emulation_results = emulation_config.read_all_emulator_groups()
    emulator_predictions_x = emulation.predict(
        x, emulation_config, emulator_results=emulation_results, analysis_settings=config.analysis_settings
    )
    emulator_predictions_x_prime = emulation.predict(
        x_prime, emulation_config, emulator_results=emulation_results, analysis_settings=config.analysis_settings
    )

    # Convert to dict: emulator_predictions[observable_label]
    observables = data_IO.read_dict_from_h5(config.output_dir, "observables.h5", verbose=False)
    emulator_predictions_x_dict = data_IO.observable_dict_from_matrix(
        emulator_predictions_x["central_value"], observables, observable_filter=emulation_config.observable_filter
    )
    emulator_predictions_x_prime_dict = data_IO.observable_dict_from_matrix(
        emulator_predictions_x_prime["central_value"], observables, observable_filter=emulation_config.observable_filter
    )

    # Construct dict of sensitivity index, in same format as emulator_predictions['central_value']
    sensitivity_index_dict = emulator_predictions_x_prime_dict["central_value"].copy()
    sorted_observable_list = data_IO.sorted_observable_list_from_dict(observables)
    for observable_label in sorted_observable_list:
        x = emulator_predictions_x_dict["central_value"][observable_label]
        x_prime = emulator_predictions_x_prime_dict["central_value"][observable_label]
        sensitivity_index_dict[observable_label] = 1 / delta * (x_prime - x) / x

    # Plot
    plot_list = [sensitivity_index_dict]
    columns = [0]
    labels = [rf"Sensitivity index at MAP, $\delta={delta}$"]
    colors = [sns.xkcd_rgb["dark sky blue"]]
    param = (
        config.analysis_config["parameterization"][config.parameterization]["names"][i_parameter][1:-1]
        .replace("{", "{{")
        .replace("}", "}}")
    )
    ylabel = rf"$S({param}, \mathcal{{O}}, \delta)$"
    # ylabel = rf'$S({param}, \mathcal{{O}}, \delta) = \frac{{1}}{{\delta}} \frac{{\mathcal{{O}}([1+\delta] {param})-\mathcal{{O}}({param})}}{{\mathcal{{O}}({param})}}$'
    filename = f"sensitivity_index_{i_parameter}.pdf"
    plot_utils.plot_observable_panels(
        plot_list,
        labels,
        colors,
        columns,
        config.analysis_settings,
        plot_dir,
        filename,
        linewidth=1,
        ymin=-5,
        ymax=5,
        ylabel=ylabel,
        plot_exp_data=False,
        bar_plot=True,
    )


# ---------------------------------------------------------------
def _running_alpha_s(
    mu_square: float | npt.NDArray[np.float64], alpha_s: float | npt.NDArray[np.float64]
) -> float | npt.NDArray[np.float64]:
    """Running alpha_s for HTL-qhat

    Extracted from MATTER:
    https://github.com/JETSCAPE/JETSCAPE/blob/935b69291f0fd319f42dc6a9fb5960a4f814e16f/src/jet/Matter.cc#L3944-L3953

    We have a separate implementation verified by the theorists:
    https://github.com/FHead/PhysicsJetScape/blob/c3c9adfeee72e1f9ce34728e174e35ca8a70065b/JetRAAPaper/26363_HPPaperPlots/QHat.h#L10-L19

    Note:
        lambda_square_QCD_HTL is determined using alpha^fix_s such that the running alpha_s
        coincide with alpha^fix_s at scale mu^2= 1 GeV^2.

    Args:
        mu_square: Virtuality of the parton.
        alpha_s: Coupling constant (here, this is will be alpha^fix_s).

    Returns:
        float: running alpha_s
    """
    if mu_square <= 1.0:
        return alpha_s

    active_flavor: Final[int] = 3
    square_lambda_QCD_HTL = np.exp(-12 * np.pi / ((33 - 2 * active_flavor) * alpha_s))
    return 12 * np.pi / ((33 - 2 * active_flavor) * np.log(mu_square / square_lambda_QCD_HTL))


# ---------------------------------------------------------------
def qhat_over_T_cubed(posterior_samples, config, T=0, E=0) -> float:
    """
    Evaluate qhat/T^3 from posterior samples of parameters,
    for fixed E and T

    See: https://github.com/FHead/PhysicsJetScape/blob/c3c9adfeee72e1f9ce34728e174e35ca8a70065b/JetRAAPaper/26363_HPPaperPlots/QHat.h#L21-L35
    (which itself is derived from the MATTER code in jetscape).

    :param 2darray parameters: posterior samples of parameters -- shape (n_samples, n_params)
    :return 1darray: qhat/T^3 -- shape (n_samples,)
    """

    if posterior_samples.ndim == 1:
        posterior_samples = np.expand_dims(posterior_samples, axis=0)

    if config.parameterization == "exponential":
        # Inputs
        alpha_s_fix = posterior_samples[:, 0]
        # Constants
        active_flavor: Final[int] = 3
        # The JETSCAPE framework calculates qhat using the gluon Casimir factor, but
        # we by convention we typically report the quark qhat value, so we need to use
        # the quark Casimir factor.
        C_a: Final[float] = 4.0 / 3.0

        # From GeneralQhatFunction
        debye_mass_square = alpha_s_fix * 4 * np.pi * np.power(T, 2.0) * (6.0 + active_flavor) / 6
        # This is the virtuality of the parton
        # See info from Abhijit here: https://jetscapeworkspace.slack.com/archives/C025X5NE9SN/p1648404101376299
        # as well as Yi's code:
        # https://github.com/FHead/PhysicsJetScape/blob/c3c9adfeee72e1f9ce34728e174e35ca8a70065b/JetRAAPaper/26363_HPPaperPlots/QHat.h#L21-L35
        scale_net = np.maximum(2 * E * T, 1.0)

        running_alpha_s = _running_alpha_s(scale_net, alpha_s_fix)
        answer = (C_a * 50.4864 / np.pi) * running_alpha_s * alpha_s_fix * np.abs(np.log(scale_net / debye_mass_square))

        # If we wanted to return just qhat (rather than qhat/T^3), we could use the following conversion:
        # return answer * 0.19732698   # 1/GeV to fm
        # qhat/T^3 is dimensionless, so we don't need to convert units
        return answer  # noqa: RET504

    msg = f"qhat_over_T_cubed not implemented for parameterization: {config.parameterization}"
    raise RuntimeError(msg)


# ---------------------------------------------------------------
def _generate_prior_samples(config, n_samples=100):
    """
    Generate samples of prior parameters

    The prior is uniform in the parameter space -- except for c1,c2,c3 it is the log that is uniform.

    :param 2darray parameters: posterior samples of parameters -- shape (n_samples, n_params)
    :return 2darray: samples -- shape (n_samples,n_params)
    """
    names = config.analysis_config["parameterization"][config.parameterization]["names"]
    parameter_min = config.analysis_config["parameterization"][config.parameterization]["min"].copy()
    parameter_max = config.analysis_config["parameterization"][config.parameterization]["max"].copy()

    # Transform c1,c2,c3 to log
    n_params = len(names)
    for i, name in enumerate(names):
        if "c_" in name:
            parameter_min[i] = np.log(parameter_min[i])
            parameter_max[i] = np.log(parameter_max[i])

    # Generate uniform samples
    rng = np.random.default_rng()
    samples = rng.uniform(parameter_min, parameter_max, (n_samples, n_params))

    # Transform log(c1,c2,c3) back to c1,c2,c3
    for i, name in enumerate(names):
        if "c_" in name:
            samples[:, i] = np.exp(samples[:, i])

    return samples
