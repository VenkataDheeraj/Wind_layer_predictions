FROM python
RUN pip install --upgrade pip
RUN pip install hcipy hdf5plugin h5py glob2 scikit-image tqdm
RUN mkdir /app
WORKDIR /app 
