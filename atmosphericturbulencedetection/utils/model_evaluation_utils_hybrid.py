# +
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import wandb
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

import sys
sys.path.insert(0, '..')

class R_Square_2:
    def __init__(self, y, y_hat, device):
        '''
        Function to calculate R-Squared value of (input image and predicted image) and (true labels and predicted labels)

        Inputs:
        x - Tensor of shape (N,C,D,H,W) denotes the input of the auto-encoder
        x_tilde - Tensor of shape (N,C,D,H,W) denotes the regenerated image of the auto-encoder
        y - Tensor of shape (N,2*WL) denotes the true wind layer labels 
        y_hat - Tensor of shape (N,2*WL) denotes the predicted wind layer labels of the Regressor

        Outputs: dictionary containing
        r_square_image - Tensor of shape(1) denotes r_squared value of the image and predicted_image
        r_square_windspeed_i - Tensor of shape(1) denotes r_squared value of wind speed of layer i
        r_square_direction_i - Tensor of shape(1) denotes r_squared value of wind direction of layer i
        r_square_mean - Tensor of shape(1) denotes MEAN r_squared value
        '''

        self.device = device
        self.n = y.shape[0]
        
        self.y_mean = self.calculate_mean(y).to(self.device)
        self.y_var = self.calculate_variance(y).to(self.device)

        self.y_error = self.calculate_mse(y,y_hat).to(self.device)
        
    def calculate_mean(self, x):
        return torch.mean(x.view(x.shape[0],-1), dim=0)
    
    def calculate_variance(self, x):
        return torch.var(x.view(x.shape[0],-1), dim=0)
    
    def adjust_mean(self, x_bar, n, y_bar, m):
        return (x_bar*n + y_bar*m) / (n+m)
    
    def adjust_var(self, n, x_bar, x_var, m, y_bar, y_var):
        
        new_var = ((n-1)*x_var + (m-1)*y_var) / (n+m-1)
        new_var += (n*m*((x_bar - y_bar)**2))/((n+m)*(n+m-1))
        
        return new_var
    def update_values(self, new_y, new_y_hat):
        
        m = new_y.shape[0]
        
        new_batch_mean = self.calculate_mean(new_y).to(self.device)
        new_batch_var = self.calculate_variance(new_y).to(self.device)
        
        new_y_var = self.adjust_var(self.n, self.y_mean, self.y_var, m, new_batch_mean, new_batch_var).to(self.device)
        new_y_mean = self.adjust_mean(self.y_mean, self.n, new_batch_mean, m).to(self.device)
        
        self.y_mean = new_y_mean
        self.y_var = new_y_var
        
        del new_batch_mean
        del new_batch_var
        
        new_batch_y_mse = self.calculate_mse(new_y, new_y_hat).to(self.device)

        new_y_mse = self.adjust_mean(self.y_error, self.n, new_batch_y_mse, m).to(self.device)
        
        self.n += m
        self.y_error = new_y_mse
        
        del new_batch_y_mse
        gc.collect()
        return
    def calculate_mse(self, x, x_hat):
        return F.mse_loss(x,x_hat, reduction='none').mean(dim=0).view(-1).to(self.device)
    
    def calculate_r2_score(self, rss, tss, eps=1e-14):
        return 1 - (rss/( tss + eps))
    def fetch_r2_score(self):
        y_tss = (self.y_var*(self.n-1)).to(self.device)
        
        y_rss = (self.y_error*self.n).to(self.device)
        
        y_dim = y_rss.shape[0]
        y_r2_score = self.calculate_r2_score(y_rss, y_tss).to(self.device)
        
        result = {}
        
        wind_layers = y_rss.shape[0]//2
        for i in range(wind_layers):
            result[f'r_square_windspeed_{i+1}'] = y_r2_score[i].item()
            result[f'r_square_direction_{i+1}'] = y_r2_score[wind_layers + i].item()
        
        y_r2_score = y_r2_score.mean()
        
        return result


# +
def initialize_test_results(test_results, batch_results):
    for k, v in batch_results.items():
        test_results[k] = 0
    
def update_test_results(test_results, batch_results):
    for k, v in batch_results.items():
        test_results[k] += v
        
def update_test_results_mean(test_results, num_batches, r2_score_results):
    for k, v in test_results.items():
        test_results[k] /= num_batches
        
    test_results.update(r2_score_results)


# -

def plot_scatter(y_true, y_pred, range_wind_speeds, min_wind_speeds, file_path='./', modulus = True):
    wind_layers = y_true.shape[1]//2
    
    y_true[:,:wind_layers] = y_true[:,:wind_layers]*range_wind_speeds[:wind_layers] + min_wind_speeds[:wind_layers]
    y_pred[:,:wind_layers] = y_pred[:,:wind_layers]*range_wind_speeds[:wind_layers] + min_wind_speeds[:wind_layers]
    if not modulus:
        y_true[:,wind_layers:] = y_true[:,wind_layers:]*360
        y_pred[:,wind_layers:] = y_pred[:,wind_layers:]*360
    else:
        y_true[:,wind_layers:] = y_true[:,wind_layers:]*180
        y_pred[:,wind_layers:] = y_pred[:,wind_layers:]*180
    

    fig, ax = plt.subplots(wind_layers, 2, figsize=(12,4*wind_layers))
    
    if wind_layers == 1:
        sns.scatterplot(ax = ax[0], x=y_true[:,0], y=y_pred[:,0])
        sns.scatterplot(ax = ax[1], x=y_true[:,1], y=y_pred[:,1])

        ax[0].set_xlabel(f'Actual Layer {1} wind speed')
        ax[1].set_xlabel(f'Actual Layer {1} wind direction')

        ax[0].set_ylabel(f'Predicted Layer {1} wind speed')
        ax[1].set_ylabel(f'Predicted Layer {1} wind direction')


        #ax[0].set_xlim(left=min_wind_speeds[0], right=min_wind_speeds[0] + range_wind_speeds[0])
        #ax[1].set_xlim(left=-5, right=(360 if not modulus else 180)+5)

        #ax[0].set_ylim(bottom=min_wind_speeds[0] - range_wind_speeds[0]*0.01, top=min_wind_speeds[0] + range_wind_speeds[0]*1.01)
        #ax[1].set_ylim(bottom=-5, top=(360 if not modulus else 180)+5)
        
    else:
        for i in range(wind_layers):
            sns.scatterplot(ax = ax[i,0], x=y_true[:,i], y=y_pred[:,i])
            sns.scatterplot(ax = ax[i,1], x=y_true[:,i+wind_layers], y=y_pred[:,i+wind_layers])

            ax[i,0].set_xlabel(f'Actual Layer {i+1} wind speed')
            ax[i,1].set_xlabel(f'Actual Layer {i+1} wind direction')

            ax[i,0].set_ylabel(f'Predicted Layer {i+1} wind speed')
            ax[i,1].set_ylabel(f'Predicted Layer {i+1} wind direction')


            #ax[i,0].set_xlim(left=min_wind_speeds[i] - 1, right=min_wind_speeds[i] + range_wind_speeds[i] + 1)
            #ax[i,1].set_xlim(left=0, right=360)

            #ax[i,0].set_ylim(bottom=min_wind_speeds[i] - 1, top=min_wind_speeds[i] + range_wind_speeds[i] + 1)
            #ax[i,1].set_ylim(bottom=0, top=360)

    plt.suptitle('Actual vs Prediction scatterplot')
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, 'scatter_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()



def run_evaluation_v2(model, device, device2, testing_data, run_name, wandb_params):
    test_results = {}
    data_load_ratio, batch_size = wandb_params['data_load_ratio'], wandb_params['test_batch_size']
    model_type, wind_layers = wandb_params['model_type'], wandb_params['wind_layers']
    folder_path = wandb_params['folder_path']
    if wandb_params['download_model'] == 1:
        folder_path = './'
    params = {'batch_size': batch_size,'shuffle': False}
    testing_generator = torch.utils.data.DataLoader(testing_data, **params)
    
    y_true = np.zeros((testing_data.file_len, 2*wind_layers))
    y_pred = np.zeros((testing_data.file_len, 2*wind_layers))
    
    with torch.no_grad(): 
        model.to(device)
        r2_scores = None
        num_batches = 0
        wandb_log_step = 0

        num_file_window = testing_data.file_len//(data_load_ratio*batch_size)
        testing_data.reset_frame_start()
        for file_window in range(num_file_window):
            frame_windows = testing_data.num_window
            total_batches = int(len(testing_data.file_list)*frame_windows/batch_size)
            rand_window = np.random.randint(0,frame_windows)
            for window in range(frame_windows):
                for gen_index, (img_data_pixel, img_data_freq, labels, zernike) in enumerate(testing_generator):
                    num_batches += 1
                    if num_batches % 100 == 0:
                        print(f'Processing batch {num_batches}/{total_batches}', flush=True)
                    img_data_pixel.unsqueeze_(1)
                    img_data_pixel = img_data_pixel.detach() 
                    img_data_freq = img_data_freq.detach()
                    labels = labels.detach()
                    if model_type in ('cnn','fno', 'unet', 'fno_unet','cnn_baseline','cnn_vicreg', 'tfno', 'spec'):
                        z, y_hat = model(img_data_pixel, img_data_freq)
                    elif model_type == 'vae':
                        x_tilde, mean, log_var, y_hat = model(img_data)
                    batch_results = {'regressor_mse': float(F.mse_loss(labels, y_hat, reduction='mean').item())}
                    if num_batches == 1:
                        r2_scores = R_Square_2(labels.detach().to(device2), y_hat.detach().to(device2), device2)
                        initialize_test_results(test_results, batch_results)
                    else:
                        r2_scores.update_values(labels.detach().to(device2), y_hat.detach().to(device2)) 
                    
                    if window == rand_window:
                        start_index = data_load_ratio*batch_size*file_window + gen_index*batch_size
                        end_index = start_index + batch_size
                        y_true[start_index:end_index] = labels.cpu().numpy()
                        y_pred[start_index:end_index] = y_hat.cpu().numpy()
                        
                    update_test_results(test_results, batch_results)
                    torch.cuda.empty_cache()
                    # End Generator
                testing_data.increment_frame_start()
                #End Windows
            testing_data.reset_frame_start()
            torch.cuda.empty_cache()
            testing_data.fetch_new_batch(batch_size*data_load_ratio)
            wandb_log_step +=1
            #End File window
        # End torch.no_grad()
    update_test_results_mean(test_results, num_batches, r2_scores.fetch_r2_score())
    np.savez(os.path.join(folder_path, f'label_predictions.npz'),compressed=True, y_true=y_true, y_pred=y_pred)
    plot_scatter(y_true, y_pred, testing_data.range_wind_speeds, testing_data.min_wind_speeds, folder_path)
    print(', '.join([f'{k}: {v}\n' for k,v in test_results.items()]), flush=True)
    return test_results
