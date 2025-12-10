FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN pip install torchinfo hdf5plugin glob2 h5py neuraloperator wandb torch_harmonics mlflow toml seaborn

RUN mkdir /app

WORKDIR /app

EXPOSE 5000