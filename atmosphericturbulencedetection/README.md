# README #

Repository to Run Dockerised containers for Multitask Learner models for Atmospherix Trubulence Detection by wind layer prediction

### What is this repository for? ###

* The repository runs Multi-task models using Resnet based regressors to predict wind layers
* Version 3

### How do I get set up? ###
* Clone the repository
* Create docker-compose.yaml file in the parent directory
* Paste the following code

```yaml

version: "3.8"
services:
  hcipy_speckle:
    container_name: sdapse_hcipy_speckle
    build: 
        context: ./hcipy_speckle/
        dockerfile: datagen.dockerfile
    volumes:
      - ./hcipy_speckle/:/app
      - /data/SDAPSE/python_test_data/:/app/dataset/
    ipc: host
    tty: true
    stdin_open: true
    command: 'python main.py --wind-file=/app/Wind_Files/wind_profile_data_layer_4_test_data.txt --max-workers=16'
  multitask_0:
    container_name: sdapse_multitask_0
    build: ./AtmosphericTurbulenceDetection/
    volumes:
      - ./AtmosphericTurbulenceDetection/:/app
      - /data/SDAPSE/:/data/SDAPSE/
    ipc: host
    tty: true
    stdin_open: true
    environment:
      - WANDB_API_KEY=<WANDB_API_KEY>
      - WANDB_PROJECT=<WANDB_PROJECT>
      - WANDB_ENTITY=<WANDB_ENTITY>
    command: 'python train.py --ngpu 0 --batch-num 1'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  model_eval_multitask_0:
    build: ./AtmosphericTurbulenceDetection/
    volumes:
      - ./AtmosphericTurbulenceDetection/:/app
      - /data/SDAPSE/:/data/SDAPSE/
    ipc: host
    tty: true
    stdin_open: true
    environment:
      - WANDB_API_KEY=<WANDB_API_KEY>
      - WANDB_PROJECT=<WANDB_PROJECT>
      - WANDB_ENTITY=<WANDB_ENTITY>
    command: 'python model_eval.py --ngpu 0'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```


* change `--max-workers=25` to `--max-workers=X` where X is the number of processors for multi-processing
* change `wind_profile_data_batch_5.txt` to `wind_profile_data_batch_X.txt` where X is the number of the data batch
* Run `docker-compose up --build hcipy_speckle` to start the data generator container
* Run `docker-compose up --build multitask_0` to start the training container on gpu 0
* Run `docker-compose up --build model_eval_multitask_0` to start the model evaluation container on gpu 0

### Configuration ###

* Autoencoder model types: [FNO, CNN]
* Autoencoder latent dimensions: [100, 1000]
* Autoencoder fully connected layer nodes: [4000]
* data generation modes: [frame slicing, frame skipping]
* regressor model types: [mlp, resnet]
* Number of frames: [1000]
* L1 regularization: [1e-4, 1e-5]
* modulus 180: [True, False]


### Dependencies ###
* train data saved in folder in the folder `/data/SDAPSE/python_train_data/layer_Y/batch_X`
* test data saved in folder in the folder `/data/SDAPSE/python_test_data/layer_Y`

### Who do I talk to? ###

* Repo owner or admin
* Vignesh Kumar Pandian Sathia (spandian1@gsu.edu)