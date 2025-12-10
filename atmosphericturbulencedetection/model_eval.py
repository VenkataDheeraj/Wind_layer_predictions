import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle as pkl
import wandb    
import toml
import argparse
from torchinfo import summary
import sys
sys.path.insert(0, '..')

from data_generators.data_generator_hybrid_difference import DataLoader
from utils.model_evaluation_utils import run_evaluation_v2
from utils.utils import fetch_run_name, fetch_model,fetch_loss_function, fetch_save_folder_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document helper.....')
    
    parser.add_argument('--ngpu', type=int, default=1, help='set the gpu on which the model will run')
    args = parser.parse_args()
    
    ngpu      = args.ngpu
    
    with open("test_config.toml", "r") as f:
        config = toml.load(f, _dict=dict)

    TEST_DATA_PATH = config['test_params']['data_dir']
    SAVE_FOLDER_BASE = config['test_params']['base_dir']
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{ngpu}" if (use_cuda and config['test_params']['cuda']==1)  else "cpu")
    use_cuda = False
    device2 = torch.device(f"cuda:{1-ngpu}" if (use_cuda and config['test_params']['cuda']==1)  else "cpu")
    data_load_ratio = 2
    batch_size = config['test_params']['batch_size']
    threshold = config['test_params']['threshold']
    
    run_id = config['test_params']['run_id']
    download_model = config['test_params']['download_model']

    project_name = os.getenv("WANDB_PROJECT", "SDAPSE")   
    entity_name = os.getenv("WANDB_ENTITY", "gsu-dmlab")  
    api_key = os.getenv("WANDB_API_KEY", None)            
    
    api = wandb.Api(key = api_key)
    run = api.run(run_id)
    run_name = run.name
    run_info = run_name.split('_')
    job_id = run_info[0]
    model_filename = str(job_id) + "_model_batch_1_sum_vicreg_after_training.pth"
    model_type = run_info[1]
    loss_strategy = run_info[2]




    latent_dim = int(run_info[5])
    frames = int(run_info[7])
    datagen_mode = run_info[8]
    batch_num = int(run_info[12])
    regressor_type = run_info[10]
    
    
    print(run_id)
    wandb.init(project=project_name, id=run.id, resume="must", allow_val_change=True)
    
    wandb_params = run.config
    # Adding code copy to wandb as artifact
    code_artifact = wandb.Artifact(name=f"{job_id}_code_snapshot", type="code")
    code_artifact.add_dir('/data/SDAPSE/code/SDAPSE')
    wandb.log_artifact(code_artifact)
    
    
    wind_layers = int(wandb_params['wind_layers'])
    fc2_nodes = int(wandb_params['fc2_nodes'])
    
    
    wandb_params['data_load_ratio'] = data_load_ratio
    wandb_params['test_batch_size'] = batch_size
    wandb_params['download_model'] = download_model
    loss_strategy = wandb_params['loss_function'].lower()
    
    model = "_".join(run_info[1:])
    MODEL_FILEPATH = os.path.join(SAVE_FOLDER_BASE, job_id, model_filename)
    print(MODEL_FILEPATH)
    
    model = fetch_model(model_type,fc2_nodes, latent_dim, device, frames, wind_layers, loss_strategy, domain)
    
    if download_model == 1:
        wandb_params["folder_path"] = f"{SAVE_FOLDER_BASE}/{job_id}"
        model_path = f"{wandb_params['folder_path']}/{model_filename}"
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
        else:
            artifact = wandb.use_artifact(f"{entity_name}/{project_name}/{model_filename}:latest", type="model")
            artifact_dir = artifact.download(root=wandb_params["folder_path"])
            model_path = f"{artifact_dir}/{model_filename}"
            # Load the model
            checkpoint = torch.load(model_path, weights_only=False)
    else: 
        checkpoint = torch.load(MODEL_FILEPATH, weights_only=False)
    
    if "_metadata" in checkpoint['model_state_dict']:
        del checkpoint['model_state_dict']["_metadata"]  # Remove unwanted metadata
    model.load_state_dict(checkpoint['model_state_dict'])

    testing_data = DataLoader(TEST_DATA_PATH, wind_layers, frames, batch=batch_size*data_load_ratio, frame_type=datagen_mode, mode ='test', device=device, loss_type=loss_strategy)

    run.summary['test_batch_size'] = batch_size
    run.summary['test_mse_threshold'] = threshold
    
    test_result = run_evaluation_v2(model, device, device2, testing_data, f'{run_name}', wandb_params)
    
    scatter_plot_artifact = wandb.Artifact(f'{job_id}_scatter_plot', type='evaluation')
    scatter_plot_artifact.add_file(os.path.join(wandb_params['folder_path'], 'scatter_plot.png'))
    wandb.log_artifact(scatter_plot_artifact)

    wandb.log(test_result)
    
    wandb.summary.update(test_result)
    # Finish the run
    wandb.finish()


