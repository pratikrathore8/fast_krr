from functools import partial

from tqdm import tqdm

from constants import (
    USE_LATEX,
    FONTSIZE,
    HPARAMS_TO_LABEL,
    BASE_SAVE_DIR,
    EXTENSION,
)
from constants import ENTITY_NAME, PROJECT_LIN_CVG
from constants import LIN_CVG
from base_utils import set_fontsize, render_in_latex
from cfg_utils import get_save_dir, create_krr_config, plot_runs_dataset_grid

# use a different x-axis for linear convergence
X_AXIS = "datapasses"

# save directory
SAVE_DIR = "lin_cvg_v2"

# filters for runs
ASKOTCH_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "accelerated": lambda run: run.config["accelerated"],
    "nystrom": lambda run: run.config["precond_params"] is not None
    and run.config["precond_params"]["type"] == "nystrom",
    "damped": lambda run: run.config["precond_params"] is not None
    and run.config["precond_params"]["rho"] == "damped",
    "sampling": lambda run: run.config["sampling_method"] == "uniform",
    "precision": lambda run: run.config["precision"] == "float64",
    "finished": lambda run: run.state == "finished",
}
SAP_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "accelerated": lambda run: run.config["accelerated"],
    "newton": lambda run: run.config["precond_params"] is not None
    and run.config["precond_params"]["type"] == "newton",
    "regularization": lambda run: run.config["precond_params"] is not None
    and run.config["precond_params"]["rho"] == "regularization",
    "sampling": lambda run: run.config["sampling_method"] == "uniform",
    "precision": lambda run: run.config["precision"] == "float64",
    "finished": lambda run: run.state == "finished",
}


if __name__ == "__main__":
    set_fontsize(FONTSIZE)
    if USE_LATEX:
        render_in_latex()

    plot_fn = partial(
        plot_runs_dataset_grid,
        entity_name=ENTITY_NAME,
        hparams_to_label=HPARAMS_TO_LABEL,
        x_axis=X_AXIS,
        save_dir=get_save_dir(BASE_SAVE_DIR, SAVE_DIR),
        extension=EXTENSION,
        keep_largest_m_runs=False,
    )

    full_krr_cfg_float64 = create_krr_config(
        PROJECT_LIN_CVG, [ASKOTCH_FILTER, SAP_FILTER]
    )

    with tqdm(total=len(LIN_CVG), desc="Linear convergence") as pbar:
        for datasets_cfg in LIN_CVG:
            plot_fn(
                full_krr_cfg=full_krr_cfg_float64,
                inducing_krr_cfg=None,
                datasets_cfg=datasets_cfg,
                name_stem="full_",
            )
            pbar.update(1)
