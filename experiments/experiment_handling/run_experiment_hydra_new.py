import hydra
from omegaconf import DictConfig, OmegaConf

from experiment import Experiment
from experiment_handling.utils import (
    validate_experiment_args,
    set_precision,
    set_random_seed,
)


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    """
    Main function to run the experiment using Hydra.
    """
    # Print the configuration for debugging
    print(OmegaConf.to_yaml(cfg))

    # Set precision and random seed
    set_precision(cfg.training.precision)
    set_random_seed(cfg.training.seed)

    # Convert kernel and preconditioner params to mutable dicts
    kernel_params = OmegaConf.to_container(cfg.kernel, resolve=True)
    precond_params = OmegaConf.to_container(cfg.precond, resolve=True)

    # Organize arguments for the experiment
    experiment_args = {
        "dataset": cfg.dataset,
        "model": cfg.model,
        "task": cfg.task,
        "kernel_params": kernel_params,
        "lambd_unscaled": cfg.lambd_unscaled,
        "opt": cfg.opt.type,
        "precond_params": precond_params
        if precond_params.get("type", None) is not None
        else None,
        "log_freq": cfg.training.log_freq,
        "log_test_only": cfg.training.log_test_only,
        "precision": cfg.training.precision,
        "seed": cfg.training.seed,
        "device": f"cuda:{cfg.device}",
        "wandb_project": cfg.wandb.project,
    }

    # Optional arguments
    if cfg.training.max_iter is not None:
        experiment_args["max_iter"] = cfg.training.max_iter
    if cfg.training.max_time is not None:
        experiment_args["max_time"] = cfg.training.max_time

    # Add model-specific arguments
    if cfg.model == "inducing_krr":
        experiment_args["m"] = cfg.m

    # Add optimizer-specific arguments
    if cfg.opt.type == "askotchv2":
        experiment_args["block_sz_frac"] = cfg.opt.block_sz_frac
        experiment_args["sampling_method"] = cfg.opt.sampling_method
        experiment_args["mu"] = cfg.opt.mu
        experiment_args["nu"] = cfg.opt.nu
        experiment_args["accelerated"] = cfg.opt.accelerated
    if cfg.opt.type == "askotchv3":
        experiment_args["block_sz_frac"] = cfg.opt.block_sz_frac
        experiment_args["sampling_method"] = cfg.opt.sampling_method
        experiment_args["accelerated"] = cfg.opt.accelerated
    if cfg.opt.type in ["eigenpro2", "eigenpro3"]:
        experiment_args["block_sz"] = cfg.opt.block_sz
        experiment_args["r"] = cfg.opt.r
        experiment_args["bg"] = cfg.opt.bg
        if cfg.opt.type == "eigenpro2":
            experiment_args["gamma"] = cfg.opt.gamma
        if cfg.opt.type == "eigenpro3":
            experiment_args["proj_inner_iters"] = cfg.opt.proj_inner_iters
    if cfg.opt.type == "mimosa":
        experiment_args["bg"] = cfg.opt.bg
        experiment_args["bH"] = cfg.opt.bH
        experiment_args["bH2"] = cfg.opt.bH2

    # Validate the experiment arguments
    validate_experiment_args(experiment_args)
    print(experiment_args)

    # Run the experiment
    exp = Experiment(experiment_args)
    exp.run()


if __name__ == "__main__":
    main()
