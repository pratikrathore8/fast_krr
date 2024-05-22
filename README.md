# Fast KRR

<img src="logo.webp" width="400" height="400" alt="SKOTCH Logo">

Companion code for "Have ASkotch: Fast Methods for Large-scale, Memory-constrained Kernel Ridge Regression".

# Recreating our environment

Please create a Python virtual environment and activate it. After activation, please run `pip install -r requirements.txt`.

## Obtaining the Taxi dataset

Please clone [this GitHub repo](https://anonymous.4open.science/r/nyc-taxi-data). Then run `filter_runs.py` and `yellow_taxi_processing.sh` (NOTE: you may have to turn off the move to Google Drive step in this script) in this repo.

This will genrerate a `.h5py` file for each month from January 2009 to December 2015. Move these files to a folder `data/taxi-data` and run `taxi_processing.py` in this repo.

## Obtaining the HOMO dataset

## Obtaining the SUSY and HIGGS datasets

Please run `download_data.py`.

## Running the experiments

For each dataset, please run the corresponding `run_all.sh` file in the `config` folder. For example, the experiments for HOMO can be run using `./config/homo/run_all.sh`.

## Plotting
