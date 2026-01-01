# %% 
### init
###
import os
import cv2  
import torch
import numpy as np
import utils.utils as utils
import matplotlib.pyplot as plt
from utils.gen_pattern import pattern_selector
from utils.load_config import load_config, update_config, convert_config
from optimize import init_model,e2e_opt,fab_opt,height_opt,fab_eval

dtype = torch.float32
opt = load_config('./config', 'base.yml') #load basic config

# load config files
# opt=update_config(opt,load_config('./config', 'thu_2step.yml'))
# opt=update_config(opt,load_config('./config', 'gauss_2step.yml'))
opt=update_config(opt,load_config('./config', 'fresnel_lens_2step.yml'))
# opt=update_config(opt,load_config('./config', 'grating_2step.yml'))
# opt=update_config(opt,load_config('./config', 'splitter_2step.yml'))
# opt=update_config(opt,load_config('./config', 'splitter_2D_2step.yml'))
# opt=update_config(opt,load_config('./config', 'dot_generator_2step.yml'))

opt = convert_config(opt)

device = torch.device(opt.device)

propagator, fab_function = init_model(opt,device=device) # init physical models

###### Gen. Target Pattern
opt, target_amp = pattern_selector(opt,plot=True,
                    device=device,dtype=dtype)

filename = os.path.join('fig', opt[0].name + '_Target')
utils.plot_result(target_amp,dpi=2000,filename=filename)

#### 2-step optimize   
######### 1:phase opt
# batch_idx = range(0,4)
batch_idx = None
# single step
max_loss = float('inf')
max_loop = 1
# # multiple step
# max_loss = 0.0
# max_loop = 5
# for p in opt:  
#     p.method_phase = 'BS'
#     # p.method_phase = 'GS'
#     p.optim_params_phase.num_iters = 300
#     # p.method_phase = 'SGD'
#     # p.optim_params_phase.num_iters = 8000
plot_log_scale = False
s0 = (opt[0].slm_res[0]/opt[0].roi_res[0] *\
     opt[0].slm_res[1]/opt[0].roi_res[1])**0.5
depth_max = float(fab_function.gt(torch.tensor([0.0],device=device)))
height_2pi = float(opt[0].wavelength[0]/(opt[0].refraction_index[0]-1))
space = depth_max - height_2pi
height_min = space/2
height_phase, recon_amp = height_opt(opt,target_amp,
                                propagator,
                                height_2pi, 
                                s0,
                                plot_result = True, 
                                plot_log_scale=plot_log_scale,
                                output = True,
                                batch_idx = batch_idx,
                                max_loss = max_loss,
                                max_loop = max_loop,
                                device=device)
height_phase[0] = height_phase[0] + height_min
utils.plot_result([height_phase[0]],['Height'])


filename = os.path.join('fig', opt[0].name + '_Recon')
utils.plot_result(recon_amp,dpi=2000,filename=filename)

# %%
#### 2-step optimize   
######### 2:fab opt
batch_idx = None
# single step
max_loss = float('inf')
max_loop = 1
# max_loss = 0.0
# max_loop = 10
for p in opt:  
    # p.method = 'BS'
    # p.optim_params.num_iters = 350
    p.method = 'SGD'
    # p.loss_params.custom = 'L1'
    # p.optim_params.lr = 5.0e+2
    p.loss_params.custom = None  
    p.optim_params.lr = 1.0e+2
    p.optim_params.num_iters = 20000
    p.optim_params.eps = 1.0e-8
#     p.optim_params.amsgrad = False

dose_out_8bit, height_out, recon_amp = fab_opt(opt,height_phase,
                                propagator, fab_function,
                                start_point='inverse',
                                plot_result = True,
                                plot_log_scale=plot_log_scale,
                                output = True,
                                batch_idx = batch_idx,
                                max_loss = max_loss,
                                max_loop = max_loop,
                                device=device)
# save pt files
filename = os.path.join(opt[0].root_path, opt[0].run_id+'_phase.pt')
torch.save(height_phase, filename)    
filename = os.path.join(opt[0].root_path, opt[0].run_id+'.pt')
torch.save(dose_out_8bit, filename)  
     
# %% #### 2-step optimize   
######### 2: direct fab with GT
opt[0].name = '0609_fres_direct'
opt[0].run_id = opt[0].name
final_dose = torch.clamp(fab_function.gt_inv(height_phase[0]),0.0,255.0)
final_dose_8bit = final_dose.to(torch.uint8)
final_dose_8bit = final_dose_8bit.squeeze().cpu().detach().numpy()
if final_dose_8bit.ndim ==1:
    final_dose_8bit = final_dose_8bit[:,None]
dose_out_8bit = [final_dose_8bit]
# save BMP
if opt[0].save_wrap_res is not None:
    dose_save = utils.pad_image(final_dose_8bit,
                        opt[0].save_wrap_res,
                        pytorch=False,
                        mode='wrap')
else:
    dose_save = final_dose_8bit
if opt[0].save_pixel_multiply is not None:
    # change pixel size
    dose_save = np.kron(dose_save,np.ones((opt[0].save_pixel_multiply,opt[0].save_pixel_multiply)))
filename = os.path.join(opt[0].root_path, opt[0].run_id+'.bmp')
cv2.imwrite(filename, dose_save.astype(np.uint8))
print(f'    - Done, Save: ',filename)
# %%
######### eval

# log_scale=False
log_scale=True
pytorch=True
db_min=-50

dose_out_eval = dose_out_8bit[0]
target_amp_eval = target_amp[0]

recon_field, recon_phase = fab_eval(opt[0], dose_out_eval, target_amp_eval,
                prop_method='FFT', 
                normalize=True, device=device) 
recon_spectrum = recon_field.abs()**2
# calc. ratio
cutoff_frequency = 0.2
_, _, h, w = recon_spectrum.shape
fy = int((cutoff_frequency*h)//2)
fx = int((cutoff_frequency*w)//2)
mask = torch.zeros_like(recon_spectrum)
mask[..., h//2-fy:h//2+fy, w//2-fx:w//2+fx] = 1
print('LF ratio: ', torch.sum(mask*recon_spectrum)/torch.sum(recon_spectrum))

recon_field, _ = fab_eval(opt[0], dose_out_eval, target_amp_eval,
                prop_method='ASM', 
                output_size=[0.002,0.002],
                normalize=True, device=device) 
recon_intensity = recon_field.abs()**2
# calc. ratio
cutoff_frequency = 0.10
_, _, h, w = recon_intensity.shape
fy = int((cutoff_frequency*h)//2)
fx = int((cutoff_frequency*w)//2)
mask = torch.zeros_like(recon_intensity)
mask[..., h//2-fy:h//2+fy, w//2-fx:w//2+fx] = 1
print('SPOT ratio: ', torch.sum(mask*recon_intensity)/torch.sum(recon_intensity))

# plot
utils.plot_result([dose_out_eval],['Dose'],
                pytorch=False,
                dpi=500,normalize=False,
                log_scale=False)    
utils.plot_result([recon_phase],['Phase'],
                pytorch=pytorch,
                dpi=500,normalize=False,
                log_scale=False)
utils.plot_result([recon_spectrum],['Spectrum'],
                pytorch=pytorch,
                dpi=500,normalize=False,
                log_scale=log_scale,db_min=db_min)
utils.plot_result([recon_intensity],['Recon.'],
                pytorch=pytorch,
                dpi=500,normalize=False,
                log_scale=log_scale,db_min=db_min)
utils.plot_result([target_amp_eval**2],['Target'], 
                    dpi=500,normalize=False,
                    log_scale=log_scale,db_min=db_min)

# %%
#### load bmp file 
from PIL import Image
import cv2
data_path = 'Z:\\BMP\\20230221'
filename = '0221_grating_e2e.bmp'
save_wrap_res = [12000,12000]
im = Image.open(os.path.join(data_path,filename))
# convert to uint8
im = np.array(im.convert('L'), dtype=np.uint8)
dose_out_8bit = [im]
dose_save = utils.pad_image(im,
                    save_wrap_res,
                    pytorch=False,
                    mode='wrap')
# save & replace
cv2.imwrite(os.path.join(data_path,filename), dose_save.astype(np.uint8))
print(f'    - Done, Save: ',filename)
# %%
#### load bmp file
from PIL import Image
from torchvision import transforms
data_path = 'Z:\\BMP\\20230217'
filename = '0221_grating_direct.bmp'
im = Image.open(os.path.join(data_path,filename))
# convert to uint8
im = np.array(im.convert('L'), dtype=np.uint8)
dose_out_8bit = [im]
# %%
#### load and save flipped checkpoints files
import re
idx = 0
# filename = opt[idx].load_checkpoint_file
filename = opt[idx].save_checkpoint_file

x,s = utils.load_checkpoints(filename)
x = x.clone().detach().flip(2)

filename = re.sub(r'_(\d+)_(\d+)', r'_\1_-\2', filename)
utils.save_checkpoints(filename, x, s)
print(filename,': Saved') 

# %% 
###############################
###############################
# Merge, repeat and save batches

# repeat times in y and x
# repeats = (1,1) # do not repeat
repeats = (1,40)
# repeats = (1,110)

if 'dose_out_8bit' in locals():
    dose_load = dose_out_8bit
else:
    batch = [str(s) for s in range(len(opt))]
    dose_load = list()
    for i,b in enumerate(batch):
        print('Load: ',opt[i].root_path, opt[i].run_id+'_'+b+'.bmp')
        filename = os.path.join(opt[i].root_path, opt[i].run_id+'_'+b+'.bmp')
        dose = cv2.imread(filename, 0)[None,None,:,:]
        if opt[i].save_wrap_res[0] > dose.shape[-2] or opt[i].save_wrap_res[1] > dose.shape[-1]:
            dose = utils.pad_image(dose, opt[i].save_wrap_res,
                            pytorch=False,mode='wrap')
        dose_load.append(dose)
        
    dose_load = np.concatenate(dose_load,axis=0)

# merge as plot_layout
dose_load = np.expand_dims(dose_load, axis=1)
dose_load = utils.tile_batch(dose_load,
                                opt[0].plot_layout,
                                pytorch=False,
                                uint8=True).squeeze().astype(np.uint8)
# dose_load = np.kron(dose_load, np.ones(repeat_size)) # change pixel size
dose_load = np.tile(dose_load, repeats)
filename = os.path.join(opt[0].root_path, opt[0].run_id+'_Merged'+'.bmp')
cv2.imwrite(filename, dose_load)

print(f'    - Done, Save: ',filename)  
# %%
