from functools import partial

import numpy as np
from tqdm import tqdm

from constants import ENTITY_NAME, PROJECT_FULL_KRR, PROJECT_INDUCING_KRR
from constants import FONTSIZE, USE_LATEX, BASE_SAVE_DIR, EXTENSION
from constants import (
    PERFORMANCE_DATASETS_CLASSIFICATION_CFG,
    PERFORMANCE_DATASETS_REGRESSION_CFG,
)
from base_utils import (
    set_fontsize,
    render_in_latex,
    _get_clean_data,
    get_x,
    plot_performance_grid,
)
from cfg_utils import create_krr_config, _get_filtered_runs, get_save_dir

# save directory
SAVE_DIR = "performance_comparison_v2"

# tolerances for the performance comparison
CLASSIFICATION_EPS = 1e-2  # 1e-3
REGRESSION_REL_EPS = 1e-2

# scaled time budget for the performance comparison
# changing the third argument of np.linspace changes the resolution of the time budget
TIME_BUDGET = np.linspace(0, 1, 101)

ASKOTCH_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "accelerated": lambda run: run.config["accelerated"],
    "preconditioned": lambda run: run.config["precond_params"] is not None,
    "rho_damped": lambda run: run.config.get("precond_params", {}).get("rho", None)
    == "damped",
    "sampling": lambda run: run.config["sampling_method"] == "uniform",
    "block_sz_frac": lambda run: run.config["block_sz_frac"] == 0.01,
    "finished": lambda run: run.state == "finished",
}
EIGENPRO2_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "eigenpro2",
    "finished": lambda run: run.state == "finished",
}
EIGENPRO3_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "eigenpro3",
    "finished": lambda run: run.state == "finished",
}
PCG_FLOAT32_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float32",
    "not_greedy_cholesky_or_falkon": lambda run: not (
        run.config["precond_params"]["type"] == "partial_cholesky"
        and run.config["precond_params"]["mode"] == "greedy"
    )
    and not run.config["precond_params"]["type"] == "falkon",
    "finished": lambda run: run.state == "finished",
}
FALKON_FLOAT32_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float32",
    "falkon": lambda run: run.config["precond_params"]["type"] == "falkon",
    "finished": lambda run: run.state == "finished",
}
PCG_FLOAT64_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float64",
    "not_greedy_cholesky_or_falkon": lambda run: not (
        run.config["precond_params"]["type"] == "partial_cholesky"
        and run.config["precond_params"]["mode"] == "greedy"
    )
    and not run.config["precond_params"]["type"] == "falkon",
    "finished": lambda run: run.state == "finished",
}
FALKON_FLOAT64_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "pcg",
    "precision": lambda run: run.config["precision"] == "float64",
    "falkon": lambda run: run.config["precond_params"]["type"] == "falkon",
    "finished": lambda run: run.state == "finished",
}
FILTERS_BASE = {
    "askotchv2": ASKOTCH_FILTER,
    "eigenpro2": EIGENPRO2_FILTER,
    "eigenpro3": EIGENPRO3_FILTER,
}
FILTERS_FLOAT32 = {
    **FILTERS_BASE,
    "pcg": PCG_FLOAT32_FILTER,
    "falkon": FALKON_FLOAT32_FILTER,
}
FILTERS_FLOAT64 = {
    **FILTERS_BASE,
    "pcg": PCG_FLOAT64_FILTER,
    "falkon": FALKON_FLOAT64_FILTER,
}


def get_krr_runs(dataset_cfgs, krr_configs):
    krr_runs = {}
    for dataset_cfg in dataset_cfgs:
        for ds in dataset_cfg["datasets"].keys():
            krr_runs[ds] = {}
            krr_runs[ds]["runs"] = {}
            krr_runs[ds]["metric"] = dataset_cfg["datasets"][ds]["metric"]
            for key, value in krr_configs.items():
                krr_runs[ds]["runs"][key] = _get_filtered_runs(value, ds, ENTITY_NAME)
    return krr_runs


def get_peak_metric(run, metric):
    y, _ = _get_clean_data(run, metric)
    if metric == "test_acc":
        return y.max()
    else:
        return y.min()


def get_best_metric_val(krr_runs_dataset):
    metric = krr_runs_dataset["metric"]
    best_metric_val = 0 if metric == "test_acc" else np.inf
    for _, value in krr_runs_dataset["runs"].items():
        for run in value:
            peak_metric = get_peak_metric(run, metric)
            if metric == "test_acc":
                best_metric_val = max(best_metric_val, peak_metric)
            else:
                best_metric_val = min(best_metric_val, peak_metric)
    return best_metric_val


def get_stopping_point(run, metric, best_val):
    y, steps = _get_clean_data(run, metric)
    times = get_x(run, steps, "time")
    if metric == "test_acc":
        eps = CLASSIFICATION_EPS
        relative = False
    else:
        eps = REGRESSION_REL_EPS
        relative = True

    if relative:
        condition = y <= best_val * (1 + eps)
    else:
        condition = y >= best_val - eps

    indices = np.where(condition)[0]
    if indices.size > 0:
        return times[indices[0]]
    return np.inf


def convert_time_to_frac(time, run):
    return time / run.config["max_time"]


def get_best_stopping_points(krr_runs_dataset, best_metric_val_dataset):
    metric = krr_runs_dataset["metric"]
    best_stopping_points = {}
    for key, value in krr_runs_dataset["runs"].items():
        best_stopping_point = np.inf
        for run in value:
            stopping_point = get_stopping_point(run, metric, best_metric_val_dataset)
            best_stopping_point = min(best_stopping_point, stopping_point)
        # Implicitly assumes all runs for the same optimizer have the same time budget
        if len(value) > 0:
            best_stopping_point_relative = convert_time_to_frac(
                best_stopping_point, value[0]
            )
        else:
            best_stopping_point_relative = np.inf
        best_stopping_points[key] = {
            "absolute": best_stopping_point,
            "relative": best_stopping_point_relative,
        }
    return best_stopping_points


def get_best_stopping_points_all(dataset_cfgs, krr_configs):
    # Get all runs for the KRR configurations across the datasets
    krr_runs = get_krr_runs(dataset_cfgs, krr_configs)

    # Get the point where we reach the best metric value w.r.t. the stopping criterion
    best_stopping_points_all = {}
    for ds, runs in krr_runs.items():
        print(f"Processing dataset: {ds}")
        best_metric_vals_ds = get_best_metric_val(runs)
        best_stopping_points_all[ds] = get_best_stopping_points(
            runs, best_metric_vals_ds
        )

    return best_stopping_points_all


def get_scaled_performance(best_stopping_points_all, time_budget):
    # Loop through the datasets and get performance for each optimizer
    performance = {}
    for _, best_stopping_points_ds in best_stopping_points_all.items():
        for opt, best_stopping_point in best_stopping_points_ds.items():
            relative_stopping_point = best_stopping_point["relative"]
            if opt not in performance:
                performance[opt] = np.zeros_like(time_budget)
            performance[opt] += np.where(time_budget >= relative_stopping_point, 1, 0)

    # Scale the performance by the number of datasets
    for opt, perf in performance.items():
        performance[opt] = perf / len(best_stopping_points_all)

    return performance


def get_scaled_performance_precision(
    filters, datasets_classification_cfg, datasets_regression_cfg, time_budget
):
    krr_configs = {}
    for key, value in filters.items():
        if key in ["eigenpro3", "falkon"]:
            krr_configs[key] = create_krr_config(
                proj_name=PROJECT_INDUCING_KRR, base_criteria=[value]
            )
        else:
            krr_configs[key] = create_krr_config(
                proj_name=PROJECT_FULL_KRR, base_criteria=[value]
            )

    # Get the "stopping" points based on the best metric value for each dataset
    best_stopping_points_classification = get_best_stopping_points_all(
        datasets_classification_cfg, krr_configs
    )
    # best_stopping_points_regression = get_best_stopping_points_all(
    #     datasets_regression_cfg, krr_configs
    # )

    # Convert the stopping points to performance
    performance_classification = get_scaled_performance(
        best_stopping_points_classification, time_budget
    )
    # performance_regression = get_scaled_performance(
    #     best_stopping_points_regression, time_budget
    # )

    # return performance_classification, performance_regression
    return performance_classification


if __name__ == "__main__":
    set_fontsize(FONTSIZE)
    if USE_LATEX:
        render_in_latex()

    performance_fn = partial(
        get_scaled_performance_precision,
        datasets_classification_cfg=PERFORMANCE_DATASETS_CLASSIFICATION_CFG,
        datasets_regression_cfg=PERFORMANCE_DATASETS_REGRESSION_CFG,
        time_budget=TIME_BUDGET,
    )

    # plot_fn = partial(
    #     plot_performance_grid,
    #     x_vals=TIME_BUDGET,
    #     titles=["Classification", "Regression"],
    #     n_cols=2,
    #     n_rows=1,
    #     save_dir=get_save_dir(BASE_SAVE_DIR, SAVE_DIR),
    # )

    plot_fn = partial(
        plot_performance_grid,
        x_vals=TIME_BUDGET,
        titles=["Classification"],
        n_cols=1,
        n_rows=1,
        save_dir=get_save_dir(BASE_SAVE_DIR, SAVE_DIR),
    )

    with tqdm(total=2, desc="Performance overall") as pbar:
        # perf_classification_float32, perf_regression_float32 = performance_fn(
        #     filters=FILTERS_FLOAT32
        # )
        perf_classification_float32 = performance_fn(filters=FILTERS_FLOAT32)
        plot_fn(
            # performance_dicts=[perf_classification_float32, perf_regression_float32],
            performance_dicts=[perf_classification_float32],
            save_name=f"float32_overall.{EXTENSION}",
        )
        pbar.update(1)
        # perf_classification_float64, perf_regression_float64 = performance_fn(
        #     filters=FILTERS_FLOAT64
        # )
        perf_classification_float64 = performance_fn(filters=FILTERS_FLOAT64)
        plot_fn(
            performance_dicts=[perf_classification_float64],
            save_name=f"float64_overall.{EXTENSION}",
        )
        pbar.update(1)
