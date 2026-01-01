# Function to load yaml configuration file
import os,yaml
import math
import numpy as np
import torch
from easydict import EasyDict as edict
from datetime import datetime

def load_config(config_path,config_name):
    with open(os.path.join(config_path, config_name)) as file:
        cfg = edict(yaml.safe_load(file))
    
    # print optimizer & propogation model
    print('Config file: ,',config_name,' loaded...')
    
    # path setting
    if 'name' in cfg:
        cfg.run_id = f'{cfg.name}' 
    
    return cfg

def update_config(cfg,B):
    # update nested dict cfg by B
    for key, value in B.items():
        if key in cfg:
            if isinstance(value, dict):
                update_config(cfg[key], value)
            else:
                cfg[key] = value
        else:
            cfg[key] = value

    return cfg

def convert_config(cfg):
    # data type conversion
    # numpy array
    cfg.refraction_index = np.array(cfg.refraction_index).reshape([1,-1,1,1])
    cfg.wavelength = np.array(cfg.wavelength).reshape([1,-1,1,1])
    cfg.prop_dist = np.array(cfg.prop_dist).reshape([1,-1,1,1])
    # torch tensor
    device = torch.device(cfg.device)
    cfg.loss_params.channel_weight = torch.tensor(cfg.loss_params.channel_weight,device=device).reshape([1,-1,1,1])
    cfg.loss_params.batch_weight = torch.tensor(cfg.loss_params.batch_weight,device=device).reshape([-1,1,1,1])
    cfg.loss_params_phase.channel_weight = torch.tensor(cfg.loss_params.channel_weight,device=device).reshape([1,-1,1,1])
    cfg.loss_params_phase.batch_weight = torch.tensor(cfg.loss_params.batch_weight,device=device).reshape([-1,1,1,1])
    return cfg

def show_config(cfg):
    # print('Wavelengths (nm) ',cfg.wavelength)
    print('Pixel Size (um) ',cfg.feature_size)
    print('Prop. Model: ',cfg.prop_model)
    # if cfg.output_size != None:
    #     print('Used Diff. Angle: ', round(math.degrees(math.atan(max(cfg.output_size)/2/max(cfg.prop_dist))),2))
    #     print('Max. Diff. Angle: ', round(math.degrees(math.asin(min(cfg.wavelength)/max(cfg.feature_size))),2))
    # elif cfg.prop_model != 'FFT':
    #     print('Used Diff. Angle: ', round(math.degrees(math.atan(max(cfg.output_res)*max(cfg.feature_size)/max(cfg.prop_dist))),2))
    #     print('Max. Diff. Angle: ', round(math.degrees(math.asin(min(cfg.wavelength)/max(cfg.feature_size))),2))