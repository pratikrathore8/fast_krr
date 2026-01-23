import math
import os
import re
import warnings

import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from constants import (
    BLKSZ_LABEL,
    DUMMY_PLOTTING_RANK,
    LEGEND_SPECS,
    MARKERSIZE,
    METRIC_AX_PLOT_FNS,
    METRIC_LABELS,
    MODE_LABELS,
    NAN_REPLACEMENT,
    NORM,
    OPT_COLORS,
    OPT_LABELS,
    PERFORMANCE_AXIS_LABELS,
    PRECOND_LABELS,
    PRECOND_MARKERS,
    RANK_LABEL,
    RHO_LABEL,
    RHO_LABELS,
    SAMPLING_LABELS,
    SAMPLING_LINESTYLES,
    SORT_KEYS,
    SZ_COL,
    SZ_ROW,
    TOT_MARKERS,
    X_AXIS_LABELS,
    X_AXIS_TIME_GRACE,
)
from get_opt import _get_opt
from sorting import sort_data


def set_fontsize(fontsize):
    plt.rcParams.update({"font.size": fontsize})


def render_in_latex():
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def get_project_runs(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    return runs


def check_criteria(run, criteria):
    for _, criterion in criteria.items():
        if not criterion(run):
            return False
    return True


def filter_runs(runs, criteria):
    return [run for run in runs if check_criteria(run, criteria)]


def filter_runs_union(runs, criteria_list):
    runs_filtered = []
    for criteria in criteria_list:
        runs_filtered.extend(filter_runs(runs, criteria))
    return runs_filtered


def _get_datapasses(run, steps):
    opt = run.config["opt"]
    n = run.config["n"]
    m = run.config.get("m", None)
    bg = run.config.get("bg", None)
    block_sz = run.config.get("block_sz", None)
    scaling_factor = None

    if opt == "askotchv2":
        scaling_factor = block_sz / n
    elif opt == "pcg":
        if run.config["precond_params"]["type"] == "falkon":
            scaling_factor = (2 * m * n + m**2) / (n**2)
        else:
            scaling_factor = 1
    elif opt == "mimosa":
        scaling_factor = 2 * m * bg / (n**2)
    # TODO(pratik): add scaling factors for other optimizers

    return scaling_factor * steps


def _get_cum_times(run):
    times_hist = run.scan_history(keys=["iter_time"])
    times = np.array([time["iter_time"] for time in times_hist])
    cum_times = np.cumsum(times)
    return cum_times


def _get_avg_time(run):
    cum_times = _get_cum_times(run)
    return cum_times[-1] / cum_times.shape[0]


def get_x(run, steps, x_axis):
    if x_axis == "time":
        cum_times = _get_cum_times(run)
        return cum_times[steps]
    elif x_axis == "datapasses":
        return _get_datapasses(run, steps)
    elif x_axis == "iters":
        return steps


def _get_blksz(run):
    return run.config.get("block_sz", None)


def _get_rank(run):
    if run.config["precond_params"] is not None:
        return run.config["precond_params"].get("r", None)
    return None


def _rank_label(run):
    r = _get_rank(run)
    if r is not None:
        return f"${RANK_LABEL} = {r}$"
    return None


def _blksz_label(run):
    blksz = _get_blksz(run)
    if blksz is not None:
        return f"${BLKSZ_LABEL} = {blksz}$"
    return None


def _precond_label(run):
    if run.config["precond_params"] is not None:
        precond_type = run.config["precond_params"]["type"]
        label_comps = PRECOND_LABELS[precond_type].copy()
        if precond_type == "nystrom":
            run_rho = run.config["precond_params"]["rho"]
            label_comps.append(f"${RHO_LABEL} = {RHO_LABELS.get(run_rho, run_rho)}$")
        if precond_type == "partial_cholesky":
            label_comps.append(f"{MODE_LABELS[run.config['precond_params']['mode']]}")
        return ", ".join(label_comps)
    return None


def _sampling_label(run):
    if "sampling_method" in run.config:
        return SAMPLING_LABELS[run.config["sampling_method"]]
    return None


def _inducing_label(run):
    if "m" in run.config:
        return f"$m = {run.config['m']}$"
    return None


LABEL_FNS = {
    "r": _rank_label,
    "precond": _precond_label,
    "sampling_method": _sampling_label,
    "m": _inducing_label,
    "b": _blksz_label,
}


def get_label(run, hparams_to_label):
    opt = _get_opt(run)
    hparam_labels = [OPT_LABELS[opt]]
    for hparam in hparams_to_label:
        if hparam in ["r", "m"]:
            hparam_label = LABEL_FNS[hparam](run)
            if hparam_label is not None:
                hparam_labels.append(hparam_label)
    return ", ".join(hparam_labels)


def get_color(opt, rank):
    if isinstance(OPT_COLORS[opt], Colormap):
        return OPT_COLORS[opt](NORM(rank))
    else:
        return OPT_COLORS[opt]


def get_style(run, n_points):
    style = {}

    opt = _get_opt(run)
    if opt in ["askotchv2", "skotchv2", "nsap", "sap"]:
        style["linestyle"] = SAMPLING_LINESTYLES[run.config["sampling_method"]]

    r = _get_rank(run) if opt not in ["nsap", "sap"] else _get_blksz(run)
    r_adj = 1 if r is None else r + 1

    if run.config["precond_params"] is not None:
        precond_type = run.config["precond_params"]["type"]
        if precond_type == "nystrom":
            style["marker"] = PRECOND_MARKERS[precond_type].get(
                run.config["precond_params"]["rho"], "h"
            )
        elif precond_type == "partial_cholesky":
            style["marker"] = PRECOND_MARKERS[precond_type][
                run.config["precond_params"]["mode"]
            ]
        elif precond_type == "falkon":
            style["marker"] = PRECOND_MARKERS[precond_type][run.config["m"]]
            r_adj = DUMMY_PLOTTING_RANK + 1
    if opt == "eigenpro3":
        style["marker"] = PRECOND_MARKERS["falkon"][run.config["m"]]
        r_adj = DUMMY_PLOTTING_RANK + 1

    style["color"] = get_color(opt, r_adj)

    style["markevery"] = math.ceil(n_points / TOT_MARKERS)
    style["markersize"] = MARKERSIZE
    return style


def get_n_sci(run):
    n = run.config["n"]
    n_sci = re.sub(r"e\+?0*(\d+)", r" \\cdot 10^{\1}", f"{n:.2e}")
    n_sci = re.sub(r"e-0*(\d+)", r" \\cdot 10^{-\1}", n_sci)
    return n_sci


def _detect_nans(y):
    nan_index = np.where(np.isnan(y))[0]
    if nan_index.size > 0:
        first_nan_index = nan_index[0]
        return first_nan_index
    return None


def _clean_data(y):
    # If there are NaNs in y, set all elements from the first NaN onwards to infinity
    first_nan_index = _detect_nans(y)
    if first_nan_index is not None:
        y[first_nan_index:] = NAN_REPLACEMENT
    return y


def _get_clean_data(run, metric):
    y_hist = run.scan_history(keys=[metric, "_step"])
    y = np.array([hist[metric] for hist in y_hist], dtype=np.float64)
    steps = np.array([hist["_step"] for hist in y_hist])
    return _clean_data(y), steps


def _plot_run(run, metric, x_axis, hparams_to_label, plot_fn):
    y, steps = _get_clean_data(run, metric)
    x = get_x(run, steps, x_axis)
    label = get_label(run, hparams_to_label[_get_opt(run)])
    style = get_style(run, y.shape[0])

    (handle,) = plot_fn(x, y, label=label, **style)
    return label, handle


def keep_largest_m(runs_ds, metric):
    best_runs = {}

    for run in runs_ds:
        opt = _get_opt(run)
        m_value = run.config.get("m", None)
        if m_value is None:
            continue

        # for eigenpro3, only keep runs that have no NaNs in the metric
        if opt == "eigenpro3":
            y, _ = _get_clean_data(run, metric)
            if np.any(y == NAN_REPLACEMENT):
                continue

        if opt not in best_runs or m_value > best_runs[opt].config["m"]:
            best_runs[opt] = run

    return list(best_runs.values())


def get_save_path(save_dir, save_name):
    if save_dir is not None and save_name is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return os.path.join(save_dir, save_name)
    elif save_dir is not None and save_name is None:
        warnings.warn(
            "Must provide save_name if save_dir is provided. Plot will not be saved."
        )
        return None
    elif save_dir is None and save_name is not None:
        warnings.warn(
            "Must provide save_dir if save_name is provided. Plot will not be saved."
        )
        return None
    else:
        return None


def plot_runs_axis(
    ax,
    run_list,
    hparams_to_label,
    metric,
    plot_fn_str,
    x_axis,
    xlim,
    ylim,
    title,
):
    # Allow overriding the default plot function for the metric
    if plot_fn_str is not None:
        plot_fn = getattr(ax, plot_fn_str)
    else:
        plot_fn = getattr(ax, METRIC_AX_PLOT_FNS[metric])
    # Sort the data so runs appear on top of each other in a consistent order
    run_list = sort_data(run_list, sort_keys=SORT_KEYS)
    labels = {}

    for run in run_list:
        label, handle = _plot_run(run, metric, x_axis, hparams_to_label, plot_fn)
        labels[run] = {"label": label, "handle": handle}

    # Set the x-axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    # If no explicit x-axis limits are provided are plotting with
    # respect to time, restrict the x-axis to the maximum time
    elif x_axis == "time":
        max_time = run_list[0].config["max_time"]
        ax.set_xlim(0, max_time * X_AXIS_TIME_GRACE)

    n_sci = get_n_sci(run_list[0])
    ax.set_ylim(ylim)
    ax.set_title(f"{title} ($n = {n_sci}$)")
    ax.set_xlabel(X_AXIS_LABELS[x_axis])
    ax.set_ylabel(METRIC_LABELS[metric])
    return labels


def plot_runs_grid(
    run_lists,
    hparams_to_label,
    metrics,
    plot_fns,
    x_axis,
    xlims,
    ylims,
    titles,
    n_cols,
    n_rows,
    save_dir=None,
    save_name=None,
):
    if x_axis not in ["time", "datapasses", "iters"]:
        raise ValueError(f"Unsupported value of x_axis: {x_axis}")
    save_path = get_save_path(save_dir, save_name)
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False, figsize=(SZ_COL * n_cols, SZ_ROW * n_rows)
    )
    axes = axes.flatten()

    labels = {}
    for i, (run_list, metric, plot_fn, xlim, ylim, title) in enumerate(
        zip(run_lists, metrics, plot_fns, xlims, ylims, titles)
    ):
        labels_subplot = plot_runs_axis(
            axes[i],
            run_list,
            hparams_to_label,
            metric,
            plot_fn,
            x_axis,
            xlim,
            ylim,
            title,
        )
        labels.update(labels_subplot)

    # Get sorting for all runs -- this is essential for sorting the legend
    all_runs = [run for run_list in run_lists for run in run_list]
    all_runs = sort_data(all_runs, sort_keys=SORT_KEYS)

    # Go through all runs and get the labels in the same order as the sorted runs
    unique_labels = {}
    for run in all_runs:
        if run in labels:
            label = labels[run]["label"]
            handle = labels[run]["handle"]
            if label not in unique_labels:
                unique_labels[label] = handle

    # Set the global legend
    fig.legend(unique_labels.values(), unique_labels.keys(), **LEGEND_SPECS)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


def plot_performance_grid(
    performance_dicts,
    x_vals,
    titles,
    n_cols,
    n_rows,
    save_dir=None,
    save_name=None,
):
    save_path = get_save_path(save_dir, save_name)
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False, figsize=(SZ_COL * n_cols, SZ_ROW * n_rows)
    )
    axes = axes.flatten()

    for i, (performance_dict, title) in enumerate(zip(performance_dicts, titles)):
        ax = axes[i]
        for opt, performance in performance_dict.items():
            ax.plot(
                x_vals,
                performance,
                label=OPT_LABELS[opt],
                color=get_color(opt, DUMMY_PLOTTING_RANK),
            )
        ax.set_title(title)
        ax.set_xlabel(PERFORMANCE_AXIS_LABELS["x"])
        ax.set_ylabel(PERFORMANCE_AXIS_LABELS["y"])

    # Collect all handles and labels
    handles, labels = [], []
    for ax in axes:
        h_ax, l_ax = ax.get_legend_handles_labels()
        for handle, label in zip(h_ax, l_ax):
            if label not in labels:  # Avoid duplicate labels
                labels.append(label)
                handles.append(handle)

    # Set the global legend
    fig.legend(handles, labels, **LEGEND_SPECS)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close(fig)


def plot_per_iter_runtime_ratios(
    run_pairs,
    x_label,
    save_dir=None,
    save_name=None,
):
    save_path = get_save_path(save_dir, save_name)
    avg_time_ratios = []

    fig, ax = plt.subplots(figsize=(SZ_COL, SZ_ROW))
    for run_pair in run_pairs:
        # Plot the minimum average time for each pair of runs
        min_avg_time_x = np.inf
        min_avg_time_y = np.inf
        for run in run_pair["x"]:
            avg_time = _get_avg_time(run)
            min_avg_time_x = min(min_avg_time_x, avg_time)
        for run in run_pair["y"]:
            avg_time = _get_avg_time(run)
            min_avg_time_y = min(min_avg_time_y, avg_time)
        time_ratio = min_avg_time_x / min_avg_time_y
        # Check that time ratio is not infinite
        if not np.isinf(time_ratio):
            avg_time_ratios.append(time_ratio)

    ax.hist(avg_time_ratios, bins=20, color="blue", alpha=1.0)
    ax.set_xlabel(x_label)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.close(fig)
