# %%
import os
import pickle
import torch
import numpy as np
import utils.utils as utils
import matplotlib.pyplot as plt
from utils.gen_pattern import pattern_selector
from utils.load_config import load_config, update_config, convert_config
from optimize import init_model,e2e_opt,fab_opt,phase_opt,fab_eval

filepath = '\\\\TOWER\\Optics\\BMP\\20230412'
name = '0412_1_2d3_650_e2e'
log_scale=False
db_min=-50

pt_filename = os.path.join(filepath,name)+ '.pt'
pkl_filename = os.path.join(filepath,name)+ '.pkl'
opt = load_config('./config', 'base.yml') #load basic config
with open( pkl_filename, 'rb') as file: # update from loaded pkl file
    opt = update_config(opt,pickle.load(file)[0])
dose = torch.load(pt_filename)[0]

opt = convert_config(opt)
device = torch.device(opt.device)
dtype = torch.float32
propagator, fab_function = init_model(opt,device=device) # init physical models
opt, target_amp = pattern_selector(opt,plot=True,
                    device=device,dtype=dtype)
opt = opt[0]
target_amp = target_amp[0]
opt.prop_model = 'None'
recon_height = fab_eval(opt,dose,
                None, fab_function,
                opt.output_size,opt.output_res,
                target_field=target_amp,
                normalize=False,
                to_8bit=False,
                device=device) 

opt.prop_model = 'FFT'
recon_field = fab_eval(opt,dose,
                propagator, fab_function,
                opt.output_size,opt.output_res,
                target_field=target_amp,
                normalize=True,
                to_8bit=False,
                device=device) 
recon_intensity = recon_field.abs()**2
# plot
utils.plot_result([dose],['Dose'],
                pytorch=False,
                dpi=500,normalize=False,
                log_scale=False)    
utils.plot_result([recon_height],['Height'],
                pytorch=True,
                dpi=500,normalize=False,
                log_scale=False)
utils.plot_result([recon_intensity],['Recon.'],
                pytorch=True,
                dpi=500,normalize=False,
                log_scale=log_scale,db_min=db_min)
utils.plot_result([target_amp**2],['Target'], 
                    dpi=500,normalize=False,
                    log_scale=log_scale,db_min=db_min)


utils.plot_result_1d(recon_intensity.ravel(),pytorch=True,type='bar')
# %%
