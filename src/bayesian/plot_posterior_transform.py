"""Plot posterior samples in unit and transformed physical parameter spaces."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from bayesian.mc_sampling.base import MCConfig

logger = logging.getLogger(__name__)


def plot(chain: npt.NDArray[np.float64], config: MCConfig, random_seed: int | None = None) -> None:
    """Plot posterior samples in unit and physical parameter spaces if configured."""
    transform_config = _transform_config(config)
    if not transform_config:
        logger.info("No physical posterior transform configured. Skipping transformed posterior plots.")
        return

    samples_unit = chain.reshape((chain.shape[0] * chain.shape[1], chain.shape[2]))
    samples_physical = transform_samples(samples_unit, transform_config, direction="forward")

    plot_dir = Path(config.output_dir) / "plot_transform"
    plot_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        Path(config.output_dir) / "posterior_unit_and_physical.npz",
        posterior_unit=samples_unit,
        posterior_physical=samples_physical,
    )
    _write_summary(
        samples_unit=samples_unit,
        samples_physical=samples_physical,
        unit_file_names=_unit_file_names(config, transform_config),
        unit_plot_labels=_unit_plot_labels(config, transform_config),
        component_names=_component_names(transform_config),
        output_path=Path(config.output_dir) / "posterior_transform_summary.txt",
    )

    plot_seed = _normalize_random_seed(random_seed)
    unit_file_names = _unit_file_names(config, transform_config)
    unit_plot_labels = _unit_plot_labels(config, transform_config)
    physical_names = _component_names(transform_config)
    axis_scales = _axis_scales(transform_config)

    for i, (file_name, label) in enumerate(zip(unit_file_names, unit_plot_labels, strict=True)):
        _save_1d_hist(
            values=samples_unit[:, i],
            output_path=plot_dir / f"posterior_unit_{file_name}",
            xlabel=label,
            title=f"Posterior in Unit Space: {label}",
            axis_scale="linear",
        )

    if samples_unit.shape[1] == 2:
        _save_2d_scatter(
            x=samples_unit[:, 0],
            y=samples_unit[:, 1],
            output_path=plot_dir / f"posterior_unit_{unit_file_names[0]}_{unit_file_names[1]}_2d",
            xlabel=unit_plot_labels[0],
            ylabel=unit_plot_labels[1],
            title="Posterior in Unit Space (2D)",
            x_scale="linear",
            y_scale="linear",
            random_seed=plot_seed,
        )

    for i, (name, axis_scale) in enumerate(zip(physical_names, axis_scales, strict=True)):
        _save_1d_hist(
            values=samples_physical[:, i],
            output_path=plot_dir / f"posterior_physical_{name}",
            xlabel=name,
            title=f"Transformed Posterior: {name}",
            axis_scale=axis_scale,
        )
        if axis_scale != "linear":
            _save_1d_hist(
                values=samples_physical[:, i],
                output_path=plot_dir / f"posterior_physical_{name}_linear",
                xlabel=name,
                title=f"Transformed Posterior: {name} (linear axis)",
                axis_scale="linear",
            )

    if samples_physical.shape[1] == 2:
        output_stem = f"posterior_physical_{physical_names[0]}_{physical_names[1]}_2d"
        _save_2d_scatter(
            x=samples_physical[:, 0],
            y=samples_physical[:, 1],
            output_path=plot_dir / output_stem,
            xlabel=physical_names[0],
            ylabel=physical_names[1],
            title="Transformed Posterior in Physical Space (2D)",
            x_scale=axis_scales[0],
            y_scale=axis_scales[1],
            random_seed=plot_seed,
        )
        if any(scale != "linear" for scale in axis_scales):
            _save_2d_scatter(
                x=samples_physical[:, 0],
                y=samples_physical[:, 1],
                output_path=plot_dir / f"{output_stem}_linear",
                xlabel=physical_names[0],
                ylabel=physical_names[1],
                title="Transformed Posterior in Physical Space (2D, linear axes)",
                x_scale="linear",
                y_scale="linear",
                random_seed=plot_seed,
            )


def transform_samples(
    samples: npt.NDArray[np.float64], transform_config: dict[str, Any], direction: str = "forward"
) -> npt.NDArray[np.float64]:
    """Transform posterior samples component-wise using declarative step lists."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        msg = f"Expected samples to have shape (n_samples, n_parameters), got {samples.shape}"
        raise ValueError(msg)

    components = transform_config.get("components", [])
    if samples.shape[1] != len(components):
        msg = f"Transform config has {len(components)} components, but samples have {samples.shape[1]} columns"
        raise ValueError(msg)

    transformed_components: list[npt.NDArray[np.float64]] = []
    step_key = "steps" if direction == "forward" else "inverse_steps"
    for i, component in enumerate(components):
        if step_key not in component:
            msg = f"Transform component '{component.get('name', i)}' is missing '{step_key}'"
            raise ValueError(msg)
        values = samples[:, i].copy()
        for step in component[step_key]:
            values = _apply_step(values, step)
        transformed_components.append(values)

    return np.column_stack(transformed_components)


def _apply_step(values: npt.NDArray[np.float64], step: dict[str, Any]) -> npt.NDArray[np.float64]:
    kind = step["kind"]
    if kind == "add":
        return values + float(step["value"])
    if kind == "subtract":
        return values - float(step["value"])
    if kind == "multiply":
        return values * float(step["value"])
    if kind == "divide":
        return values / float(step["value"])
    if kind == "power":
        return np.power(values, float(step["value"]))
    if kind == "tan":
        return np.tan(values)
    if kind == "arctan":
        return np.arctan(values)
    if kind == "exp":
        return np.exp(values)
    if kind == "log":
        return np.log(values)
    msg = f"Unknown transform step kind: {kind}"
    raise ValueError(msg)


def _transform_config(config: MCConfig) -> dict[str, Any] | None:
    param_cfg = config.analysis_config["parameterization"][config.parameterization]
    transform_config = param_cfg.get("physical_posterior_transform")
    if transform_config is None:
        return None
    return transform_config


def _component_names(transform_config: dict[str, Any]) -> list[str]:
    return [_slugify_component_name(component["name"]) for component in transform_config["components"]]


def _component_plot_labels(transform_config: dict[str, Any]) -> list[str]:
    return [str(component["name"]) for component in transform_config["components"]]


def _unit_plot_labels(config: MCConfig, transform_config: dict[str, Any]) -> list[str]:
    unit_labels = transform_config.get("unit_labels")
    if unit_labels is not None:
        return [str(label) for label in unit_labels]
    return [str(name) for name in config.analysis_config["parameterization"][config.parameterization]["names"]]


def _unit_file_names(config: MCConfig, transform_config: dict[str, Any]) -> list[str]:
    unit_names = transform_config.get("unit_names")
    if unit_names is not None:
        return [_slugify_component_name(name) for name in unit_names]
    n_dim = len(config.analysis_config["parameterization"][config.parameterization]["names"])
    return [f"u{i + 1}" for i in range(n_dim)]


def _axis_scales(transform_config: dict[str, Any]) -> list[str]:
    return [str(component.get("axis_scale", "linear")) for component in transform_config["components"]]


def _save_1d_hist(
    values: npt.NDArray[np.float64], output_path: Path, xlabel: str, title: str, axis_scale: str = "linear"
) -> None:
    values = _values_for_axis_scale(values, axis_scale, output_path.name)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=80, density=True, alpha=0.85, color="#1f77b4")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    if axis_scale != "linear":
        ax.set_xscale(axis_scale)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def _save_2d_scatter(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    output_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    x_scale: str = "linear",
    y_scale: str = "linear",
    n_plot_samples: int = 50000,
    random_seed: int = 12345,
) -> None:
    x, y = _paired_values_for_axis_scale(x, y, x_scale, y_scale, output_path.name)
    n_available = x.shape[0]
    n_to_plot = min(n_available, n_plot_samples)
    rng = np.random.default_rng(random_seed)
    selected = rng.choice(n_available, size=n_to_plot, replace=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        x[selected],
        y[selected],
        s=4,
        alpha=0.18,
        linewidths=0.0,
        color="#1f77b4",
        rasterized=True,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if x_scale != "linear":
        ax.set_xscale(x_scale)
    if y_scale != "linear":
        ax.set_yscale(y_scale)
    ax.text(
        0.98,
        0.02,
        f"{n_to_plot:,} samples shown",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def _write_summary(
    samples_unit: npt.NDArray[np.float64],
    samples_physical: npt.NDArray[np.float64],
    unit_file_names: list[str],
    unit_plot_labels: list[str],
    component_names: list[str],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write("Posterior transform summary\n")
        stream.write("===========================\n\n")
        for i, (name, label) in enumerate(zip(unit_file_names, unit_plot_labels, strict=True)):
            quantiles = np.quantile(samples_unit[:, i], [0.16, 0.5, 0.84])
            stream.write(f"unit {name} ({label}) q16/q50/q84: {quantiles.tolist()}\n")
        stream.write("\n")
        for i, name in enumerate(component_names):
            quantiles = np.quantile(samples_physical[:, i], [0.05, 0.16, 0.5, 0.84, 0.95, 0.99])
            stream.write(f"physical {name} q05/q16/q50/q84/q95/q99: {quantiles.tolist()}\n")


def _slugify_component_name(name: str) -> str:
    stripped = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    return stripped or "parameter"


def _normalize_random_seed(random_seed: int | np.integer[Any] | None) -> int:
    if random_seed is None:
        return 12345
    seed = int(random_seed)
    if seed < 0:
        return 12345
    return seed


def _values_for_axis_scale(
    values: npt.NDArray[np.float64], axis_scale: str, context: str
) -> npt.NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if axis_scale == "log":
        mask &= values > 0
    filtered = values[mask]
    if filtered.size == 0:
        msg = f"No valid values remain for {context} after applying axis scale '{axis_scale}'"
        raise ValueError(msg)
    if filtered.size != values.size:
        logger.warning("Dropped %d invalid samples for %s (%s axis)", values.size - filtered.size, context, axis_scale)
    return filtered


def _paired_values_for_axis_scale(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x_scale: str,
    y_scale: str,
    context: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if x_scale == "log":
        mask &= x > 0
    if y_scale == "log":
        mask &= y > 0
    filtered_x = x[mask]
    filtered_y = y[mask]
    if filtered_x.size == 0:
        msg = f"No valid samples remain for {context} after applying axis scales '{x_scale}', '{y_scale}'"
        raise ValueError(msg)
    if filtered_x.size != x.size:
        logger.warning(
            "Dropped %d invalid paired samples for %s (%s/%s axes)",
            x.size - filtered_x.size,
            context,
            x_scale,
            y_scale,
        )
    return filtered_x, filtered_y
