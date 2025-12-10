import torch
import glob2 as glob
import os
import h5py
import hdf5plugin
import numpy as np
import random
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor


# +
class DataLoader(torch.utils.data.Dataset):
    def __init__(self, folder_path, num_layers= 4, frames = 100, batch=10, frame_type='slice', mode ='train', device = torch.device('cpu'), loss_type='mse', data_normalization = 'modified_frame_difference', target = [1]*8):
        
        self.file_list = [f for f in glob.glob(os.path.join(folder_path, '*.h5'))]
        random.shuffle(self.file_list)
        self.file_len = len(self.file_list) - (len(self.file_list)%batch)
        self.file_list = self.file_list[:self.file_len]
        
        self.mode = mode
        self.frames = frames
        self.frame_type = frame_type
        self.num_layers = num_layers
        self.device = device
        self.loss_type = loss_type
        
        self.data_normalization = data_normalization
        self.target = target
        self.direction_modulus = True
        self.max_layers = 4
        self.frame_start = 0
        self.window_idx = 0
        self.h5_batch_start = 0
        
        if frames == 100:
            self.gap = 10
        else:
            self.gap = int(frames//8)
        
        # for hawaii location
        #self.min_wind_speeds = np.array([3,  7.96929180e-01, 9.93247651e+00, 1.44378574e-06])
        #self.max_wind_speeds = np.array([7, 14.93935495, 34.49111224, 12.57276457])
        #self.range_wind_speeds = self.max_wind_speeds - self.min_wind_speeds

        # for antofagasta location
        #self.min_wind_speeds = np.array([2,  7.96929180e-01, 9.93247651e+00, 1.44378574e-06])
        #self.max_wind_speeds = np.array([5, 14.93935495, 34.49111224, 12.57276457])
        #self.range_wind_speeds = self.max_wind_speeds - self.min_wind_speeds
        
        # for flagstaff location
        #self.min_wind_speeds = np.array([0,  7.96929180e-01, 9.93247651e+00, 1.44378574e-06])
        #self.max_wind_speeds = np.array([9, 14.93935495, 34.49111224, 12.57276457])
        #self.range_wind_speeds = self.max_wind_speeds - self.min_wind_speeds
        
        # for oakland location
        #self.min_wind_speeds = np.array([3,  7.96929180e-01, 9.93247651e+00, 1.44378574e-06])
        #self.max_wind_speeds = np.array([11, 14.93935495, 34.49111224, 12.57276457])
        #self.range_wind_speeds = self.max_wind_speeds - self.min_wind_speeds
        
        # for san diego location
        #self.min_wind_speeds = np.array([0,  7.96929180e-01, 9.93247651e+00, 1.44378574e-06])
        #self.max_wind_speeds = np.array([7, 14.93935495, 34.49111224, 12.57276457])
        #self.range_wind_speeds = self.max_wind_speeds - self.min_wind_speeds
        
        # for tenerife location
        #self.min_wind_speeds = np.array([2,  7.96929180e-01, 9.93247651e+00, 1.44378574e-06])
        #self.max_wind_speeds = np.array([12, 14.93935495, 34.49111224, 12.57276457])
        #self.range_wind_speeds = self.max_wind_speeds - self.min_wind_speeds
        
        # for tucson location
        #self.min_wind_speeds = np.array([0,  7.96929180e-01, 9.93247651e+00, 1.44378574e-06])
        #self.max_wind_speeds = np.array([7, 14.93935495, 34.49111224, 12.57276457])
        #self.range_wind_speeds = self.max_wind_speeds - self.min_wind_speeds
        

        # for all locations
        self.min_wind_speeds = np.array([0,  7.96929180e-01, 9.93247651e+00, 1.44378574e-06])
        self.max_wind_speeds = np.array([12, 14.93935495, 34.49111224, 12.57276457])
        self.range_wind_speeds = self.max_wind_speeds - self.min_wind_speeds
            
        self.fetch_new_batch(batch)

        self.num_frames = int(self.file_list[0].split('nframes')[1].split("_")[0]) - 1
        
        if loss_type == 'vicreg':
            self.num_window = (self.num_frames//frames - 1)*self.gap + (1 if self.mode == 'test' else 0) # VICreg
        else:
            self.num_window = (self.num_frames//frames - 1)*self.gap + 1  # MSE
            
    def normalize(self, x, x_min, x_range):
        return (x - x_min) / x_range
    
    def normalize_labels(self, labels):
        '''
        Normalize using Min-Max scaling.
        Normalize wind_directions from [0, 360] to [0,1]
        Normalize wind_speeds from [3,7](derived from global distribution) to [0,1]
        '''
        
        wind_layers = self.num_layers
        new_labels = np.zeros(2*wind_layers)
        new_labels[:wind_layers] = self.normalize(labels[:wind_layers], self.min_wind_speeds[:wind_layers], self.range_wind_speeds[:wind_layers])
        if self.direction_modulus == True:
            new_labels[wind_layers:] = (labels[self.max_layers:self.max_layers + wind_layers]/180)%1
        else:
            new_labels[wind_layers:] = labels[self.max_layers:self.max_layers + wind_layers]/360	
        return new_labels
    
    def load_h5_file(self, file_path):
        """
        Opens an HDF5 file in read mode.
        """
        return h5py.File(file_path, mode='r')
    
    def frame_difference_normalization(self):
        '''
        Normalize and CLIP values to [0,1] range
        '''
        self.h5_data = [torch.minimum((torch.abs(h5_data) / self.data_sqrt_max[i]), torch.ones_like(h5_data)) for i, h5_data in enumerate(self.h5_data)]
    
    def modified_frame_difference_normalization(self):
        '''
        Normalize to [-1,1] then [0,1] and CLIP values
        '''
        self.h5_data = [torch.minimum((h5_data / self.data_sqrt_max[i]), torch.ones_like(h5_data)) for i, h5_data in enumerate(self.h5_data)]
        self.h5_data = [torch.maximum((h5_data), -1*torch.ones_like(h5_data)) for i, h5_data in enumerate(self.h5_data)]
        self.h5_data = [(h5_data+1)/2 for i, h5_data in enumerate(self.h5_data)]
    
    def fetch_new_batch(self, batch_size = 10):
        
        self.h5_file_list = None
        self.h5_data = None
        self.h5_data_freq = None
        self.data_sqrt_max = None
        self.h5_label_string = None
        
        torch.cuda.empty_cache()
        with ThreadPoolExecutor() as executor:
            self.h5_file_list = list(executor.map(self.load_h5_file, [self.file_list[(self.h5_batch_start + i)] for i in range(batch_size)]))
        self.h5_batch_start = (self.h5_batch_start + batch_size)%len(self.file_list)
        #SQRT transform
        self.h5_data = [torch.sqrt(torch.from_numpy(h5_file.get('star_images')[:]).float().to(self.device)) for h5_file in self.h5_file_list]
        #Difference frames
        self.h5_data = [(h5_data[1:] - h5_data[:-1])for h5_data in self.h5_data]
        self.data_sqrt_max = [torch.max(torch.abs(h5_data)) for h5_data in self.h5_data]
        
        if self.data_normalization == 'frame_difference':
            self.frame_difference_normalization()
        elif self.data_normalization == 'modified_frame_difference':
            self.modified_frame_difference_normalization()
        
        
        # self.h5_data_freq = [torch.fft.fftshift(torch.fft.fftn(h5_data[:, 64:192,64:192]))[:,48:80,48:80] for h5_data in self.h5_data]  #hybrid or hybrid3
        self.h5_data_freq = [torch.fft.fftshift(torch.fft.fftn(h5_data[:, 64:192,64:192])) for h5_data in self.h5_data]   #hybrid2
        self.h5_data_freq = [torch.cat((torch.abs(h5_data_freq).view(1, *h5_data_freq.shape), ((torch.angle(h5_data_freq) + torch.pi) / (2 * torch.pi)).view(1, *h5_data_freq.shape))) for h5_data_freq in self.h5_data_freq]
        self.zernike = [torch.from_numpy(h5_file.get('zernike_coefficients')[:].T).float().to(self.device) for h5_file in self.h5_file_list]
        self.h5_label_string = [h5_file.attrs['wind_vector'] for h5_file in self.h5_file_list]
        torch.cuda.empty_cache()
    
    def increment_frame_start(self):
        self.frame_start = (self.frame_start + self.gap)% self.num_frames
    
    def reset_frame_start(self):
        self.frame_start= 0
    
    def preprocess_labels(self,label_string):
        '''
        Wrapper to handle the preprocessing of labels
        labels are in the form: [speed_1,  speed_2, speed_3, speed_4, direction_1, direction_2, direction_3, direction_4]
        '''
        normalized_labels = self.normalize_labels(np.array([float(i) for i in label_string.split(",")]))
        self.labels = torch.from_numpy(normalized_labels).float().to(self.device)
        
    def __getitem__(self, index):
        # print(f"[DEBUG] __getitem__ called with index: {index}")
        data_sqrt_max = self.data_sqrt_max[index]
        zernike = torch.zeros((1,10))
        try:
            frame_end = self.frame_start
            if self.loss_type == 'vicreg':
                frame_end += int(self.frames+(self.gap if self.mode == 'train' else 0)) #VICreg
            else:
                frame_end += int(self.frames) # MSE
            
            self.data_pixel = self.h5_data[index][self.frame_start:frame_end, 64:192,64:192]
            self.data_freq = self.h5_data_freq[index][:,self.frame_start:frame_end]
            zernike = torch.cat((self.zernike[index][frame_end-self.gap].unsqueeze(0), self.zernike[index][frame_end].unsqueeze_(0)), 0)
        except Exception as e:
            print(e, frame_end)
        self.preprocess_labels(self.h5_label_string[index])
        return self.data_pixel, self.data_freq, self.labels, zernike

    def __len__(self):
        return  len(self.h5_file_list)
# -


