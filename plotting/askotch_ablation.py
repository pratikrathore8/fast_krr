from functools import partial

from tqdm import tqdm

from constants import (
    USE_LATEX,
    FONTSIZE,
    X_AXIS,
    HPARAMS_TO_LABEL,
    BASE_SAVE_DIR,
    EXTENSION,
)
from constants import ENTITY_NAME, PROJECT_FULL_KRR
from constants import PERFORMANCE_DATASETS_CFG
from base_utils import set_fontsize, render_in_latex
from cfg_utils import get_save_dir, create_krr_config, plot_runs_dataset_grid

# save directory and filename
SAVE_DIR = "askotch_ablation"

# filters for runs
ASKOTCH_FILTER = {
    "optimizer": lambda run: run.config["opt"] == "askotchv2",
    "block_sz_frac": lambda run: run.config["block_sz_frac"] == 0.01,
    "finished": lambda run: run.state == "finished",
}

if __name__ == "__main__":
    set_fontsize(FONTSIZE)
    if USE_LATEX:
        render_in_latex()

    plot_fn = partial(
        plot_runs_dataset_grid,
        entity_name=ENTITY_NAME,
        inducing_krr_cfg=None,
        hparams_to_label=HPARAMS_TO_LABEL,
        x_axis=X_AXIS,
        save_dir=get_save_dir(BASE_SAVE_DIR, SAVE_DIR),
        extension=EXTENSION,
    )

    full_krr_cfg = create_krr_config(PROJECT_FULL_KRR, [ASKOTCH_FILTER])

    for datasets_cfg in tqdm(PERFORMANCE_DATASETS_CFG, desc="Ablation"):
        plot_fn(full_krr_cfg=full_krr_cfg, datasets_cfg=datasets_cfg, name_stem="")
