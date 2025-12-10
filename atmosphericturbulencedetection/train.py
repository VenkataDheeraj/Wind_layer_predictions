import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import toml
import os
import h5py
import hdf5plugin
import argparse
import datetime
import numpy as np
import wandb

from data_generators.data_generator_hybrid_difference import DataLoader
from utils.utils_hybrid import fetch_run_name, fetch_date_folder_name, fetch_model,fetch_loss_function, fetch_save_folder_path, load_model, save_summary, save_checkpoint, EarlyStopper, update_metrics_dict

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Document helper.....')
    
    parser.add_argument('--ngpu', type=int, default=1, help='set the gpu on which the model will run')
    parser.add_argument('--batch-num', type=int, default=1, help='set batch of the data on which model trains')
    parser.add_argument('--job-id', type=str, default='', help='set slurm job id (string)')
    args = parser.parse_args()
    
    ngpu      = args.ngpu
    batch_num = args.batch_num
    job_id = args.job_id
    
    with open("config.toml", "r") as f:
        config = toml.load(f, _dict=dict)
        
    TRAIN_DATA_BASE = config['train_params']['data_dir']         # Base path of training data
    SAVE_FOLDER_BASE = config['train_params']['base_dir']        # Base path of folder where models are saved
    
    use_cuda = torch.cuda.is_available() and config['train_params']['cuda']==1                         
    device = torch.device(f"cuda:{ngpu}" if use_cuda else "cpu")
    batch_size = config['train_params']['batch_size']
    data_load_ratio = 2                                          # File window ratio to be loaded into memory, should be integer > 1
    cur_num_epochs     = config['train_params']['cur_niter']
    retrain_num_epochs = config['train_params']['retrain_niter']
    wind_layers = config['train_params']['wind_layers']
    loss_type = config['train_params']['loss_type']              # Loss calculation, sum of losses or loss of one pathway ( AE or Regressor)
    domain = config['train_params']['domain']                    # Domain frequency or pixel
    description = config['train_params']['description']
    loss_strategy = config['train_params']['loss_strategy']      # Loss Strategy VICreg or MSE 
    l1_lambda = config['train_params']['l1_lambda']
    data_normalization = config['train_params']['data_normalization']
    
    model_type = config['model']['type']                         # Model type: [CNN, TFNO, VAE, etc]
    latent_dim = config['model']['latent_dims']                  # Number of nodes in the latent representation of AE
    fc2_nodes = config['model']['fc2_nodes']                     # Number of nodes in second dense layer
    lr = config['model']['lr']                                   # Learning rate for training
    benchmark = config['model']['benchmark']
    beta_1 = config['model']['beta_1']                           # Weight of the Autoencoder for calculating loss
    beta_2 = config['model']['beta_2']                           # Weight of the regressor for calculating loss
    load_model_filename = config['model']['load_model']          # Path of model for loading pre-trained model
    load_regressor_filename = config['model']['load_regressor']
    regressor_folder_path = config['model']['regressor_folder_path']
    

    datagen_mode = config['datagen']['mode']                     # Strategy to generate data, Could be whole data, frame slicing or skipping
    frames = config['datagen']['frames']                         # Number of frames of each data point 
    
    tfno_modes = config['model'].get('modes', [frames,32,32])
    regressor_type = config['regressor']['type']                 # Regressor model type, MLP or Resnet based
    
    cos_sim = True

    run_name = fetch_run_name(model_type, loss_strategy, domain, latent_dim, datagen_mode, regressor_type, frames, batch_num)
    
    wandb.login()
    
    cur_time = datetime.datetime.now()

    model = fetch_model(model_type, fc2_nodes, latent_dim, device, frames, wind_layers, loss_strategy, domain)

    loss_function = fetch_loss_function(model_type, loss_strategy)
    model_save_folder_path = SAVE_FOLDER_BASE
    
    if load_model_filename != '':
        load_model(model, model_save_folder_path, load_model_filename) # Load weigthts from pre-trained model

    date_folder_string = fetch_date_folder_name(cur_time)
    
    job_id = (date_folder_string if job_id == '' else job_id)
    
    if not os.path.exists(os.path.join(model_save_folder_path, job_id)):
        os.makedirs(os.path.join(model_save_folder_path, job_id))

    model_save_folder_path = os.path.join(model_save_folder_path, job_id)
    print(model_save_folder_path, flush=True)

    torch.backends.cudnn.benchmark = benchmark
    params = {'batch_size': batch_size,'shuffle': True}
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    wandb_params = {
        'model_type': model_type,
        'datagen_mode': datagen_mode,
        'latent_dims': latent_dim,
        'wind_layers': wind_layers,
        'regressor_hidden_dims': model.regressor_hidden_dims,
        'fc2_nodes': fc2_nodes,
        'regressor_type': regressor_type,
        'learning_rate': lr,
        'batch_size': batch_size,
        'current_batch_epochs': cur_num_epochs,
        # 'retrain_batch_epochs': retrain_num_epochs,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'loss_function': loss_strategy,
        'cos_sim': cos_sim,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'loss_type': 'sum' if loss_type == 0 else 'alternate',
        'datetime_created': date_folder_string,
        'description': description,
        'datetime_created': str(cur_time),
        'folder_path': model_save_folder_path,
        'domain': domain,
        'l1_lambda': l1_lambda,
        'data_normalization': data_normalization
    }
    
    wandb.init(
        project="sdapse",
        name=f'{job_id}_{run_name}',
        config=wandb_params)
    summary_path = os.path.join(model_save_folder_path, f"model_summary.txt")
    save_summary(summary_path, model, batch_size, frames, domain)
    
    artifact = wandb.Artifact(name=f"{job_id}_model_summary.txt", type="model_summary")
    artifact.add_file(summary_path)
    wandb.log_artifact(artifact)

    beta_2 = torch.tensor(beta_2, requires_grad=False).to(device)
    for cur_batch in range(batch_num, batch_num+1):
        training_data_path = os.path.join(TRAIN_DATA_BASE)
        training_data = DataLoader(training_data_path, wind_layers, frames, batch=batch_size*data_load_ratio, frame_type=datagen_mode, device=device, loss_type=loss_strategy, data_normalization=data_normalization)
        training_generator = torch.utils.data.DataLoader(training_data, **params)

        scheduler = None
        num_epochs = None
        num_file_window = len(training_data.file_list)//(data_load_ratio*batch_size)

        if cur_batch == batch_num:
            num_epochs = cur_num_epochs
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold = 0.001, threshold_mode = "rel", cooldown=1, factor=0.5, min_lr=0.000005, patience=5)
            
        else:
            num_epochs = retrain_num_epochs
            scheduler = None
        cur_loss_path = -1
        wandb_log_step = 0
        training_data.reset_frame_start()
        for epoch in range(num_epochs):
            for file_window in range(num_file_window):
                frame_windows = training_data.num_window
                first_mini_batch = True
                metrics_dict = {}
                regressor_step = 0
                ae_step = 0
                total_steps = 0 if loss_type == -1 else frame_windows
                random_loss_path = np.random.choice([0,1],frame_windows, p=[0.5,0.5])
                for window in range(frame_windows):
                    early_stopper = EarlyStopper(patience=20, min_delta=5)
                    x_tilde, y_hat = None, None
                    if loss_type == 1:
                            cur_loss_path = random_loss_path[window] # 0 AE loss 1 regressor loss
                    round_robin_step = 0
                    for img_data_pixel, img_data_freq, labels, zernike in training_generator:
                        img_data_pixel.unsqueeze_(1)         # convert dimensions from (B,D,H,W) to (B,1,D,H,W)
                        img_data_pixel = img_data_pixel.float().to(device)
                        img_data_freq = img_data_freq.float().to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        if loss_strategy == 'vicreg':
                            z_t1, y_hat_t1 = model(img_data_pixel[:,:,:frames].to(device), img_data_freq[:,:,:frames].to(device))   
                            z_t2, y_hat_t2 = model(img_data_pixel[:,:,training_data.gap:].to(device), img_data_freq[:,:,training_data.gap:].to(device))

                            loss, wind_layers_loss = loss_function(z_t1, z_t2, labels, y_hat_t1, y_hat_t2, beta_1, beta_2, cur_loss_path)
                            metrics_dict.update({
                                'predicted_t1_max_value': max(metrics_dict.get('predicted_t1_max_value',0), torch.max(z_t1).item()),
                                'predicted_t2_max_value': max(metrics_dict.get('predicted_t2_max_value',0),torch.max(z_t2).item())                                })
                        else:
                            z, y_hat = model(img_data_pixel.to(device), img_data_freq.to(device))
                            loss, wind_layers_loss = loss_function(labels, y_hat, beta_2, cur_loss_path, wind_layers, cos_sim)
                        
                        l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if 'bias' not in name)
                        loss = loss + l1_lambda*l1_norm
                        metrics_dict['lr'] = scheduler.get_last_lr()[0]
                        if cur_loss_path == -1:
                            total_steps +=1
                            metrics_dict['loss'] = metrics_dict.get('loss',0) + loss.item()
                            for i in range(wind_layers):
                                metrics_dict[f'loss_windspeed_layer_{i+1}'] = metrics_dict.get(f'loss_windspeed_layer_{i+1}',0) + wind_layers_loss[i].item()
                                metrics_dict[f'loss_wind_direction_layer_{i+1}'] = metrics_dict.get(f'loss_wind_direction_layer_{i+1}',0) + wind_layers_loss[wind_layers+i].item()

                        elif cur_loss_path == 0:
                            # AE Loss
                            ae_step +=1

                            model.handle_parameter_freezing(decoder_freeze=False, regressor_freeze=True)

                            metrics_dict['loss_ae'] = metrics_dict.get('loss_ae', 0) + loss.item()
                            cur_loss_path = 1 - cur_loss_path
                        elif cur_loss_path == 1:
                            regressor_step +=1
                            # Regressor loss
                            model.handle_parameter_freezing(decoder_freeze=True, regressor_freeze=False)
                            metrics_dict['loss_regressor'] = metrics_dict.get('loss_regressor',0) + loss.item()
                            for i in range(wind_layers):
                                metrics_dict[f'loss_windspeed_layer_{i+1}'] = metrics_dict.get(f'loss_windspeed_layer_{i+1}',0) + wind_layers_loss[i].item()
                                metrics_dict[f'loss_wind_direction_layer_{i+1}'] = metrics_dict.get(f'loss_wind_direction_layer_{i+1}',0) + wind_layers_loss[wind_layers+i].item()
                            cur_loss_path = 1 - cur_loss_path

                        loss.backward()
                        optimizer.step()
                        
                        if first_mini_batch == True:
                            print(f'Batch: {cur_batch} of {batch_num}, Epoch: {epoch+1} of {num_epochs}, File_window: {file_window} of {num_file_window}, loss: {loss.item()}', flush=True)
                            first_mini_batch = False
                        # END Training_generator
                    training_data.increment_frame_start()
                    # END all h5 files in data_load ratio
                scheduler.step(loss.item())
                metrics_dict = update_metrics_dict(metrics_dict, loss_type, ae_step, regressor_step, total_steps)
                wandb.log(metrics_dict, step = wandb_log_step)
                wandb_log_step+=1
                
                training_data.reset_frame_start()
                training_data.fetch_new_batch(batch_size*data_load_ratio)
                if file_window%100 == 0:
                    file_path = os.path.join(model_save_folder_path, f"{job_id}_model_batch_{cur_batch}_{'sum' if loss_type == 0 else 'alternate'}_vicreg_epoch_{epoch}_filewindow_{file_window}.pth")
                    save_checkpoint(model, optimizer,scheduler, epoch, scheduler.get_last_lr()[0], file_path)
                #END all files
            # END all Epochs
        model_filename = f"{job_id}_model_batch_{batch_num}_{'sum' if loss_type == 0 else 'alternate'}_vicreg_after_training.pth"
        file_path = os.path.join(model_save_folder_path, model_filename)
        save_checkpoint(model, optimizer, scheduler, epoch, scheduler.get_last_lr()[0], file_path)
        artifact_model = wandb.Artifact(name=model_filename, type="model")
        artifact_model.add_file(file_path) 
        wandb.log_artifact(artifact_model)


