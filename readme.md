README — Data-Driven Prediction of Atmospheric Wind Layers from Speckle Image Sequences

This repository contains all components required to generate synthetic atmospheric wind profiles, simulate speckle images using HCIPy, train a hybrid pixel–frequency 3D CNN model, and evaluate its performance across multiple observatory locations.

The workflow consists of three major stages:
	1.	Wind Profile Generation
	2.	Speckle Image Simulation (HCIPy) via Docker
	3.	Model Training & Evaluation (Multitask CNN) via Docker

Follow the steps below to reproduce the results in our report.


———————————————–

Wind Profile Generation

———————————————–

Navigate to:

`wind_profile_gen/`

Open the notebook:

WindProfileGen.ipynb

In the notebook you can specify:
	1. City / Observatory location
	2. Number of atmospheric layers (use 1 for this project)
	3. Number of profiles to generate
	4. Name for the file that will be generated
	
Execute all cells.
This will automatically generate a text file containing wind-profile metadata for the chosen location.


———————————————–

Data Generation

———————————————–

Move the generated profile file into:

hcipyspeckle/Wind_Files/

Edit the HCIPy simulation configuration:

Open:

docker-compose.yaml

Modify the hcipy_speckle service:

services:
  hcipy_speckle:
  	- /path/to/your/dataset/:/app/dataset/
    command: 'python main.py --wind-file=<path to wind profile file> --max-workers=16'

Set:
	•	Path to the wind-profile file
	•	Directory where speckle datasets will be generated

HCIPy simulation runs on CPU and may take several hours depending on number of profiles.

Run the simulation

`docker compose up hcipy_speckle`

This will generate .h5 speckle files under the output directory you configured.


———————————————–

Model Training

———————————————–

Navigate to:

atmosphericturbulencedetection/

(A) Configure config.toml

Set:
	•	data_dir_ → where the HCIPy .h5 files are stored
	•	base_dir_ → where model checkpoints will be saved
	•	loss_strategy → "vicreg" or "mse"
	•	batch_size, wind_layers, etc.

This file fully controls the training pipeline.

(B) Start training via Docker

`docker compose up multitask_01`

This:
	•	Loads the dataset
	•	Applies preprocessing (sqrt, frame diff, normalization, FFT)
	•	Trains the hybrid pixel–frequency 3D CNN
	•	Saves the model checkpoint to the location save_dir specified in config.toml


———————————————–

Model Evaluation

———————————————–

Edit:

test_config.toml

Set:
	•	run_id → the run_id from wandb run for the above trained model
	•	data_path → path to test dataset
	•	base_dir → output location for metrics and plots

Then run:

`docker compose up model_eval_multitask_01`

This will:
	•	Load the trained model
	•	Run evaluation on the full test set
	•	Compute R² for speed and direction
	•	Generate scatter plots and logs

Outputs are written to the directory specified in test_config.toml.


———————————————–

End-to-End Pipeline Summary

———————————————–
	1.	Generate wind profiles (Jupyter notebook)
	2.	Move generated profile file → Wind_Files/
	3.	Update docker-compose.yaml with profile path
	4.	Run HCIPy simulation → creates .h5 speckle datasets
	5.	Configure and run training via multitask_01
	6.	Configure and run evaluation via model_eval_multitask_01
	7.	Gather outputs for report (R² scores, scatter plots, checkpoints)

———————————————–

