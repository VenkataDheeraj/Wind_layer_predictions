# +
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import datetime
import sys
import math
sys.path.insert(0, '..')

from models.cnn_baseline_hybrid_domain_models import CNN_Hybrid_Domain_64, CNN_Hybrid_Domain_64_v2, CNN_Hybrid_Domain_64_v3, CNN_Hybrid_Domain_64_v2_vicreg


# -

def save_checkpoint(model, optimizer, scheduler, epoch, lr, file_path):
    scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state_dict,
            'lr': lr
            }, file_path)

# +
def off_diagonal(x):
    """ Returns the off-diagonal elements of a square matrix as a 1D tensor. """
    N, _ = x.shape
    return x.flatten()[1:].view(N - 1, N + 1)[:, :-1].flatten()

def covariance_loss(z_t1, z_t2):
    """
    Computes the covariance loss for 5D input tensors (N, C, D, H, W).
    
    Args:
        z_t1: Tensor of shape (N, C, D, H, W) or (N, D)
        x_tilde_t2: Tensor of shape (N, C, D, H, W) or (N, D)
    
    Returns:
        Scalar tensor representing the covariance loss.
    """
    
    if z_t1.dim() == 5:  # If input is 5D (N, C, D, H, W)
        N, C, D, H, W = x_tilde_t1.shape
        DHW = D * H * W
        # Flatten spatial dimensions (D, H, W) into one feature dimension
        z_t1 = z_t1.view(N, DHW)  # Flatten to (N, DHW)
        z_t2 = z_t2.view(N, DHW)  # Flatten to (N, DHW)
    
    elif z_t1.dim() == 2:  # If input is 2D (N, D)
        N, D = z_t1.shape
        DHW = D  # Feature size is D in 2D case
    else:
        raise ValueError("Input tensor must be either 2D (N, D) or 5D (N, C, D, H, W)")

    # Mean centering
    z_t1 = z_t1 - z_t1.mean(dim=0, keepdim=True)
    z_t2 = z_t2 - z_t2.mean(dim=0, keepdim=True)

    # Efficient covariance computation using einsum
    cov_z_t1 = torch.einsum('nd,ne->de', z_t1, z_t1) / (N - 1)
    cov_z_t2 = torch.einsum('nd,ne->de', z_t2, z_t2) / (N - 1)

    # Compute off-diagonal loss
    cov_loss = (off_diagonal(cov_z_t1).pow(2).sum() / DHW + off_diagonal(cov_z_t2).pow(2).sum() / DHW)

    return cov_loss

def vicreg_loss(z_t1, z_t2, lamda = 25, mu=25, nu = 1):
    
    N = z_t1.shape[0]
    
    # invariance loss
    sim_loss = F.mse_loss(z_t1, z_t2)
    
    # variance loss
    std_z_t1 = torch.sqrt(z_t1.var(dim=0) + 1e-06)
    std_z_t2 = torch.sqrt(z_t2.var(dim=0) + 1e-06)
    std_loss = torch.mean(F.relu(1 - std_z_t1)) + torch.mean(F.relu(1 - std_z_t2))
    
    # covariance loss
    cov_loss = covariance_loss(z_t1, z_t2)
    
    # loss
    loss = lamda * sim_loss + mu * std_loss + nu * cov_loss
    return loss

def vicreg_loss_function(z_t1, z_t2, y, y_hat_t1, y_hat_t2, beta_1=1, beta_2=torch.tensor([1,1,1,1,1,1,1,1], requires_grad=False), loss_path=-1):
    
    
    if loss_path == -1:
        # Sum of Losses of AE and Regressor
        sum_vicreg_loss = vicreg_loss(z_t1, z_t2) + vicreg_loss(y_hat_t1, y_hat_t2)
        regressor_loss = F.mse_loss(y, y_hat_t1, reduction='none')
        wind_layers_loss = torch.mean(regressor_loss, dim=0)
        sum_regressor_loss = torch.sum(wind_layers_loss*beta_2)
        
        calculated_loss = beta_1 * sum_vicreg_loss + sum_regressor_loss
       
    elif loss_path == 0:
        # Auto-Encoder loss only
        calculated_loss = beta_1 * (vicreg_loss(x_tilde, x_tilde_t2) + vicreg_loss(y_hat, y_hat_t2))

    else:
        # Regressor loss only
        regressor_loss = F.mse_loss(y, y_hat, reduction='none')
        wind_layers_loss = torch.mean(regressor_loss, dim=0)
        sum_regressor_loss = torch.sum(wind_layers_loss*beta_2)
        
        calculated_loss = sum_regressor_loss
    
    return calculated_loss, wind_layers_loss


# +
def mse_loss_function(y, y_hat, beta_2=torch.tensor([1,1,1,1,1,1,1,1], requires_grad=False), loss_path=-1, wind_layers=4, cos_sim = False, alpha_speed=1, alpha_dir=1):
    
    
    if loss_path == 0:
        # Auto-Encoder loss only
        non_zero = torch.count_nonzero(x)
        calculated_loss = beta_1 * (F.mse_loss(x, x_tilde, reduction='sum')/non_zero)
        return calculated_loss, torch.zeros_like(beta_2)
        
    else:    
        y_speed = y[:, :wind_layers]
        y_dir = y[:, wind_layers:]
        y_hat_speed = y_hat[:, :wind_layers]
        y_hat_dir = y_hat[:, wind_layers:]
        
        mse_speed = F.mse_loss(y_speed, y_hat_speed, reduction='none')
        mse_speed_mean = torch.mean(mse_speed, dim=0)
        speed_loss_sum = torch.sum(mse_speed_mean * beta_2[:wind_layers])
        
        if cos_sim:
            y_angle = y_dir * math.pi
            y_hat_angle = y_hat_dir * math.pi
            y_dir_vec = torch.stack([torch.cos(y_angle), torch.sin(y_angle)], dim=-1)
            y_hat_dir_vec = torch.stack([torch.cos(y_hat_angle), torch.sin(y_hat_angle)], dim=-1)
            cosine_sim = torch.sum(y_dir_vec * y_hat_dir_vec, dim=-1)
            dir_loss = 1 - cosine_sim
        else:
            dir_loss = F.mse_loss(y_dir, y_hat_dir, reduction='none')
        
        dir_loss_mean = torch.mean(dir_loss, dim=0)
        dir_loss_sum = torch.sum(dir_loss_mean * beta_2[wind_layers:])
        
        calculated_loss = alpha_speed * speed_loss_sum + alpha_dir * dir_loss_sum
        
        wind_layers_loss = torch.cat([mse_speed_mean, dir_loss_mean])
    
    return calculated_loss, wind_layers_loss


def fetch_hidden_layers(latent_dims):
    if latent_dims == 100:
        return [100,100,50,50]
    elif latent_dims == 1000:
        return [1000,500,500,200,200,100,50,50,20,10]
    else:
        print('Incorrect latent dims passed.')
        return None


def save_summary(file_path, model, batch_size, frames, domain):
    with open(file_path,'w+') as f:
        if domain in ["hybrid", "hybrid3"]:
            f.writelines(str(summary(model, input_size=[(2,1,frames,128,128), (2,2,frames,32,32)])))
        elif domain == "hybrid2":
            f.writelines(str(summary(model, input_size=[(2,1,frames,128,128), (2,2,frames,128,128)])))

def fetch_run_name(model_type, loss_strategy, domain, latent_dims, datagen_mode, regressor_type, frames, batch_num):
    name = f'{model_type}_{loss_strategy}_{domain}_latent_{latent_dims}_frames_{frames}_{datagen_mode}_regressor_{regressor_type}_batch_{batch_num}'
    return name


def fetch_save_folder_path(data_path, model_type, latent_dims, datagen_mode, wind_layers, frames):
    
    folder_string = f'{model_type}_layer_{wind_layers}_latent_{latent_dims}'
    if datagen_mode == 'slice':
        folder_string = folder_string + '_data_slice'
    elif datagen_mode == 'skip':
        folder_string = folder_string + '_skip'
    else:
        print('Incorrect datagen_mode passed.')
        return None
    model_path = os.path.join(data_path, folder_string)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path


def fetch_cnn_hybrid_domain_models(frames, fc2_nodes, latent_dims, regressor_hidden_dims, wind_layers, domain, device=torch.device('cuda:0')):
    assert frames in (8,16,32,64,100), f'Invalid number of frames, given:({frames}), allowed {8,16,32,64,100}'
    
    if domain == "hybrid":
        print("Fetching 64 frame CNN MSE Hybrid Domain model", flush = True)
        return CNN_Hybrid_Domain_64(fc2_nodes, latent_dims, regressor_hidden_dims, output_dims= 2*wind_layers, device=device)
    elif domain == "hybrid2":
        print("Fetching 64 frame CNN MSE Hybrid Domain early fusion model", flush = True)
        return CNN_Hybrid_Domain_64_v2(fc2_nodes, latent_dims, regressor_hidden_dims, output_dims= 2*wind_layers, device=device)
    elif domain == "hybrid3":
        print("Fetching 64 frame CNN MSE Hybrid Domain late fusion model", flush = True)
        return CNN_Hybrid_Domain_64_v3(fc2_nodes, latent_dims, regressor_hidden_dims, output_dims= 2*wind_layers, device=device)

def fetch_cnn_hybrid_domain_vicreg_models(frames, fc2_nodes, latent_dims, regressor_hidden_dims, wind_layers, domain, device=torch.device('cuda:0')):
    if domain == "hybrid2":
        print("Fetching 64 frame CNN VICReg Hybrid Domain early fusion model", flush = True)
        return CNN_Hybrid_Domain_64_v2_vicreg(fc2_nodes, latent_dims, regressor_hidden_dims, output_dims= 2*wind_layers, device=device)

def fetch_model(model_type, fc2_nodes, latent_dims, device, frames, wind_layers, loss_strategy, domain):
    regressor_hidden_dims = fetch_hidden_layers(latent_dims)
    
    if model_type == 'cnn':
        if loss_strategy == "mse":
            return fetch_cnn_hybrid_domain_models(frames, fc2_nodes, latent_dims, regressor_hidden_dims, wind_layers, domain, device=device)
        elif loss_strategy == "vicreg":
            return fetch_cnn_hybrid_domain_vicreg_models(frames, fc2_nodes, latent_dims, regressor_hidden_dims, wind_layers, domain, device=device)
        else:
            print('Incorrect domain')
            return None
    else:
        print('Incorrect parameters passed to fetch model')
        print(model_type, fc2_nodes, latent_dims, device, frames, wind_layers, loss_strategy, domain)
        return None


def fetch_loss_function(model_type, loss_strategy):
    if model_type in ('fno', 'cnn', 'unet', 'fno_unet', 'tfno'):
        if loss_strategy == 'vicreg':
            return vicreg_loss_function
        elif loss_strategy == 'mse':
            return mse_loss_function
    else:
        print(f'Invalid model_type in fetch_loss_function, model_type={model_type}')
        return None


def load_model(model, save_folder_path, model_filename='model_batch_1_after_training.pth'):
    '''
    Load pretrained model weights
    '''
    MODEL_FILEPATH = os.path.join(save_folder_path, model_filename)
    checkpoint = torch.load(MODEL_FILEPATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def fetch_date_folder_name(cur_datetime):
    yyyy = cur_datetime.year
    mm = cur_datetime.month
    dd = cur_datetime.day
    hh = cur_datetime.hour
    mi = cur_datetime.minute
    return f'{yyyy}{mm}{dd}{hh}{mi}'


def update_metrics_dict(metrics_dict, loss_type, ae_step, regressor_step, total_steps):
    if loss_type == 0:
        for k,v in metrics_dict.items():
            if k not in ('predicted_t1_max_value', 'predicted_t2_max_value', 'lr'):
                metrics_dict[k] = v/total_steps
        return metrics_dict
    d = {}
    for k,v in metrics_dict.items():
            if k in ('predicted_t1_max_value', 'predicted_t2_max_value', 'lr'):
                metrics_dict[k] = v
            elif k == 'loss_ae':
                metrics_dict[k] = v/ae_step
            else:
                metrics_dict[k] = v/regressor_step
    return metrics_dict

