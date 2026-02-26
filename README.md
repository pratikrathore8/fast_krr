# ASkotch: A Neat Solution for Large-scale KRR
<img src="images/logo.webp" alt="ASkotch Logo" width="200" height="200" alt="ASkotch Logo">

Companion code for [***"Have ASkotch: A Neat Solution for Large-scale Kernel Ridge Regression"***](https://arxiv.org/abs/2407.10070).
We present both a quickstart guide and detailed instructions for reproducing our experiments and figures.

## Quickstart
We present instructions for installing the repository via `pip` and using our method, `ASkotchV2`, on some example problems.

> [!NOTE]
> Currently, our implementation can only handle Laplacian, Matérn 1/2, Matérn 3/2, Matérn 5/2, and RBF kernels.
However, it is possible to add custom `KeOps`-compatible kernels by extending the `Kernel` class and modifying `kernel_inits.py`.

> [!TIP]
> Our implementation is compatible with both CPU and GPU devices.
However, we recommend using a GPU for large-scale problems.

### Installation

You can `pip` install the package using the following command:

```bash
pip install git+https://github.com/pratikrathore8/fast_krr.git
```

### R package

An R translation of this package is also available in the [`fasterKRR`](fasterKRR/) directory. See [fasterKRR-readme.md](fasterKRR-readme.md) for installation and usage instructions.

### Example usage

```python
from fast_krr.models import FullKRR
from fast_krr.opts import ASkotchV2

# Load data as PyTorch tensors
X, Xtst, y, ytst = ... # Load your data here

# Initialize the model
n = X.shape[0]
task = "classification" # Can also use "regression"
kernel_params = {"type": "rbf", "sigma": 1.0} # Can also handle Laplacian and Matérn kernels
w0 = torch.zeros(n, device=device)
lambd = 1e-6 * n

model = FullKRR(X, y, Xtst, ytst, kernel_params=kernel_params,
                 Ktr_needed=True, lambd=lambd, task=task, w0=w0, device=device)

# Initialize the optimizer
block_sz = n // 100 # Any positive integer between 1 and n
precond_type = "nystrom" # Recommended for general use
r = 100 # Any positive integer between 1 and block_sz
rho = "damped" # Recommended for general use
precond_params = {"type": precond_type, "r": r, "rho": rho}

opt = ASkotchV2(model=model, block_sz=block_sz, precond_params=precond_params)

# Train the model
max_iters = 1000

for i in range(max_iters):
    opt.step()
```

> [!NOTE]
> There is another optimizer in the package called `ASkotch`. This is an algorithm from a [previous version of our paper](https://arxiv.org/abs/2407.10070v1). We do not recommend using it over `ASkotchV2`.

### Hyperparameter recommendations
`ASkotchV2` has several hyperparameters that can be tuned to improve performance.
We describe these hyperparameters and provide suggested defaults below:
- `block_sz`: The block size. We recommend setting this to `n // 100`, where `n` is the number of training points.
- `sampling_method`: The sampling method used to select the block. The `uniform` setting is recommended for general use. The `rls` setting uses approximate leverage scores to sample the block, but it does not typically give an improvement over `uniform`.
- `precond_params`: A dictionary containing the preconditioner type and its parameters. We recommend using the Nyström preconditioner with `r = 100` and `rho = "damped"`.
- `mu`: An acceleration parameter. We recommend setting this to the regularization parameter in the `FullKRR` model that you are optimizing.
If left unspecified, `mu` will be set to the regularization parameter in the `FullKRR` model.
- `nu`: An acceleration parameter. We recommend setting this to `n`/`block_sz`.
If left unspecified, `nu` will be set to `n`/`block_sz`.
- `accelerated`: A boolean flag to enable/disable acceleration. We recommend setting this to `True`. If set to `False`, `mu` and `nu` will be ignored.

### Notebook examples

We provide two Jupyter notebooks in the `examples` folder which show how to run `ASkotchV2` for regression and classification problems.

## Instructions for reproducing our experiments and figures
Our experiments have a lot of moving parts.
Below, we provide an overview of the steps needed to reproduce our results.

### Cloning the repository
Please clone this repository using the following command:

```bash
git clone https://github.com/pratikrathore8/fast_krr.git
```

### Recreating the Python environment

> [!IMPORTANT]
> Our experiments use `Python 3.10.12` and `CUDA 12.5`. We recommend using these (or higher) Python and CUDA versions.

Please [create a virtual environment](https://docs.python.org/3/library/venv.html) and activate it. After activation, run `pip install -r requirements-dev.txt` to install all required dependencies into the virtual environment.
Finally, run `pip install -e .` in the root of this repo to install the `fast_krr` package.

### Downloading the datasets

`cd` into the `experiments` folder.
Running `download_data.py` will download all datasets we use in the paper, besides taxi.
The downloaded data will be placed in `experiments/data`.

#### Obtaining the taxi dataset

Please clone the [nyc-taxi-data repo](https://github.com/pratikrathore8/nyc-taxi-data). Run `filter_runs.py` and `yellow_taxi_processing.sh` (NOTE: you may have to turn off the move to Google Drive step in this shell script) in the nyc-taxi-data repo.

This shell script will generate a `.h5py` file for each month from January 2009 to December 2015. Move these files to a new folder `experiments/data/taxi-data`, `cd` into the `experiments` folder, and run `taxi_processing.py`.

### Running the experiments

> [!IMPORTANT]
> We log the results of our experiments using Weights & Biases.
To properly run the experiments, please create a Weights & Biases account and set up an API key.
Weights & Biases provides some [helpful documentation](https://docs.wandb.ai/quickstart/) on how to do this.

> [!IMPORTANT]
> You will have to `cd` into the `experiments` folder to run the experiments.

> [!WARNING]
> These experiments are computationally expensive and will likely take > 2 weeks to run on a single GPU.

We run the experiments in this paper by generating a large number of configurations as `.yaml` files.
To generate these configurations, run `make_configs.sh`.
This script generates the following configuration folders:
- `performance_full_krr`: Performance comparison configurations for `ASkotch` and PCG.
- `performance_inducing_krr`: Performance comparison configurations for Falkon.
- `performance_full_krr_ep2`: Performance comparison configurations for EigenPro2.
- `performance_inducing_krr_ep3`: Performance comparison configurations for EigenPro3.
- `taxi_full_krr`:
Configurations for running `ASkotch` and PCG on the taxi dataset.
- `taxi_falkon`: Configurations for running Falkon on the taxi dataset.
- `taxi_ep2`: Configurations for running EigenPro2 on the taxi dataset.
- `taxi_ep3`: Configurations for running EigenPro3 on the taxi dataset.
- `lin_cvg_full_krr`: Configurations for running linear convergence experiments with `ASkotch`.

#### Sections 7.1 and 7.4

To run the experiments in Sections 7.1 and 7.4, we will use the folders `performance_full_krr`, `performance_full_krr_ep2`, `performance_inducing_krr_ep3`, and `performance_inducing_krr`.

To run each set of experiments, run `run_experiments.py` with the appropriate configuration folder as one of the arguments. For example:

```python
python run_experiments.py --base-dir performance_full_krr --devices 0 1 --grace-period-factor 0.4
```

The argument `--devices` specifies the GPU devices to use, and `--grace-period-factor` specifies the amount of extra time to run a given experiment (since `run_experiments.py` does not account for time taken to perform inference on the test set during experiments).
Specifying multiple GPUs will run multiple experiments at the same time.
For example, `--devices 0 1` will run two experiments at the same time, one on GPU 0 and the other on GPU 1.

#### Section 7.2

To run the experiments in Section 7.2, we will use the folders `taxi_full_krr`, `taxi_falkon`, `taxi_ep2`, and `taxi_ep3`.
Similar to above, we just have to run `run_experiments.py` with the appropriate arguments.

#### Section 7.3
To run the experiments in Section 7.3, we will use the folder `lin_cvg_full_krr`.
Similar to above, we just have to run `run_experiments.py` with the appropriate arguments.

### Generating the figures

> [!IMPORTANT]
> You will have to change `ENTITY_NAME` in `plotting/constants.py` to your Weights & Biases entity name.

After running the experiments, figures can be generated by switching to the `plotting` folder and running `make_plots.sh`, which runs several Python plotting scripts in parallel.
This shell script will generate all figures in the paper.
If you would only like to generate a subset of the figures, you can run the appropriate subset of Python scripts in the `plotting` folder.

## Citation

If you find our work useful, please consider citing our paper:

```
@article{rathore2024askotch,
  title={Have ASkotch: A Neat Solution for Large-scale Kernel Ridge Regression},
  author={Pratik Rathore and Zachary Frangella and Jiaming Yang and Micha{\l} Derezi{\'n}ski and Madeleine Udell},
  journal={arXiv preprint arXiv:2407.10070},
  year={2024}
}
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
