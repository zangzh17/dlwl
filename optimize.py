# %% train

from math import inf
import numpy as np
import os
import pickle
import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import utils.utils as utils
from utils.load_config import show_config
from utils.show_results import show_results,show_batch_results
from utils.modules import SGD, GS, BS

from fit_model import fit_model,fit_pwl_model
from physical_model import propagation_ASM, propagation_FFT,propagation_SFR
from physical_model import fabrication_model

def init_model(opt,device = torch.device('cuda'), display=True):
    # show config informations
    show_config(opt)
    # Simulation models
    if opt.prop_model == 'ASM':
        propagator = propagation_ASM  # Ideal model
    elif opt.prop_model == 'FFT':
        propagator = propagation_FFT # Far-field model
    elif opt.prop_model == 'SFR':
        propagator = propagation_SFR
    elif opt.prop_model == 'None':
        propagator = None

    # initialize fabrication model
    # load MAT file
    # fit = fit_pwl_model(gt_file=opt.gt_file,lp_file=opt.lp_file,
    #                 gt_path=opt.gt_path,lp_path=opt.lp_path,
    #                 lp_ratio=opt.lp_ratio,
    #                 device=device)
    fit = fit_model(gt_file=opt.gt_file,lp_file=opt.lp_file,
                    gt_path=opt.gt_path,lp_path=opt.lp_path,
                    lp_ratio=opt.lp_ratio,
                    device=device,
                    show=display)
    fab_function = fabrication_model(fit, opt.feature_size, device)
    return propagator, fab_function

def fab_eval(opt, dose_in, target_field, 
              prop_method='None',
              output_size=[0.002,0.002],
              gt_gamma = 1.0, gt_amp=1.0, gt_clamp=True, use_gt=True,use_psf=True,
              normalize=True,
              to_8bit=False,
              device=torch.device('cpu')):
    '''
    dose_in: [B,1,H,W] numpy array or [H,W] numpy array
    target_field: [B,C,H,W] torch tensor
    '''  
    # get model
    opt.prop_model = prop_method
    opt.output_size = output_size
    propagator, fab_function = init_model(opt,device=device,display=False) # init physical models

    # input processing  
    dose_in = torch.from_numpy(dose_in).detach().to(device=device)
    if dose_in.ndim==1:
        dose_in = dose_in[None,None,:,None]
    elif dose_in.ndim==2:
        dose_in = dose_in[None,None,:,:]
    elif dose_in.ndim==3:
        dose_in = dose_in.unsqueeze(1)
    target_field = target_field.to(device=device)
    dose_in = torch.clamp(dose_in,0.0,255.0)
    if to_8bit:
        dose_in = dose_in.to(torch.uint8).to(dose_in.dtype)

    fab_function.gamma = gt_gamma
    fab_function.amp = gt_amp
    fab_function.clamp = gt_clamp
    fab_function.use_gt = use_gt
    fab_function.use_psf = use_psf

    #Eval final result
    if opt.input_power>0.0:
        phase_shift = utils.spherical_phase(dose_in.shape, 
                                            opt.feature_size, 
                                            opt.wavelength, 
                                            1.0/opt.input_power)
        phase_shift = torch.tensor(phase_shift).to(device)
    else:
        phase_shift = None
    recon_phase = utils.fabrication_field(dose_in, 
                                        wavelength=opt.wavelength, 
                                        aperture_type=opt.aperture_type,
                                        fab_model=fab_function, 
                                        fab_backward=False,
                                        refraction_index=opt.refraction_index,
                                        dtype=torch.float32)
    
    recon_field = utils.propagate_field(recon_phase, 
                        propagator, 
                        prop_dist=opt.prop_dist, 
                        wavelength=opt.wavelength, 
                        feature_size=opt.feature_size,
                        prop_model=prop_method, 
                        input_phase=phase_shift, 
                        aperture_type=opt.aperture_type,
                        zfft2=None,
                        output_size=opt.output_size,
                        output_resolution=opt.output_res,
                        normalize=normalize,
                        field_type='phase',
                        dtype=torch.float32, 
                        precomputed_H=None).cpu().detach()

    return recon_field, recon_phase

def e2e_opt(opt,target_amp_in,
             propagator, fab_function,
             plot_result = True, plot_log_scale=True,
             init_value = None,
             output = True,batch_idx=None,
             max_loss= float("inf"), max_loop=1,
             device = torch.device('cuda')):
    '''
    Optimize dose distribution for target amplitude

    Input
    ------
    opt: list
    target_amp: list
    propagator, fab_function

    Output
    ------
    dose_out
    final_recon_amp
    '''

    # Tensorboard writer
    utils.cond_mkdir(opt[0].sum_path)
    writer = SummaryWriter(opt[0].sum_path)

    
    # result dir
    utils.cond_mkdir('./checkpoints')
    utils.cond_mkdir(opt[0].root_path)
    # result numpy array
    batch_num = len(opt)
    dose_out = []
    height_out = []
    recon_amp = []

    # optimize for each batch
    if batch_idx is None:
        batch_idx = range(batch_num)
    for i in batch_idx:
        print('Batch # ', i)
        loss_value = float('inf')
        for j in range(max_loop):
            # init checkpoint/start point
            if opt[i].load_checkpoint_file is not None:
                print('Loading Checkpoints: ', opt[i].load_checkpoint_file)
                init_value,s0 = utils.load_checkpoints(opt[i].load_checkpoint_file,
                                                        device)
                
            else:
                if init_value is None:
                    if opt[i].radial_symmetry:
                        sz = max(opt[i].slm_res)//2  #select larger dimension
                        init_value = torch.clamp(opt[i].min_dose + (opt[i].max_dose-opt[i].min_dose) * torch.rand(1, 1, sz, 1),0.0, 255.0).to(device)
                    else:
                        init_value = torch.clamp(opt[i].min_dose + (opt[i].max_dose-opt[i].min_dose) * torch.rand(1, 1, *opt[i].slm_res),0.0, 255.0).to(device)
                else:
                    init_value = torch.clamp(init_value * 1.0 ,0.0 , 255.0).to(device=device)
            init_value = torch.round(init_value)
            s0 = torch.tensor(1.0,device=device)
            
            # Select Phase generation method, algorithm
            if opt[i].method == 'SGD':
                opt_algorithm = SGD(opt[i].prop_dist, 
                                    opt[i].wavelength, 
                                    opt[i].feature_size,
                                    opt[i].roi_res, 
                                    opt[i].save_checkpoint_file,
                                    opt[i].save_wrap_res,
                                    opt[i].save_pixel_multiply,
                                    phase_path=opt[i].root_path, 
                                    prop_model=opt[i].prop_model, 
                                    propagator=propagator,
                                    aperture_type=opt[i].aperture_type,
                                    radial_symmetry=opt[i].radial_symmetry,
                                    input_power=opt[i].input_power,
                                    slm_resolution=opt[i].slm_res,
                                    output_size=opt[i].output_size, 
                                    output_resolution=opt[i].output_res,
                                    fabrication=fab_function, 
                                    refraction_index=opt[i].refraction_index, 
                                    min_dose=opt[i].min_dose, max_dose=opt[i].max_dose,
                                    fab_var=opt[i].fab_var,
                                    normalize=opt[i].normalize,
                                    loss_params=opt[i].loss_params, 
                                    optim_params=opt[i].optim_params,
                                    s0=s0,
                                    writer=writer, device=device)
            elif opt[i].method == 'GS':
                opt_algorithm = GS(opt[i].prop_dist, opt[i].wavelength, 
                                    opt[i].feature_size, 
                                    opt[i].optim_params, 
                                    roi_res=opt[i].roi_res,
                                    phase_path=opt[i].root_path, 
                                    prop_model=opt[i].prop_model, 
                                    propagator=propagator,
                                    aperture_type=opt[i].aperture_type,
                                    input_power=opt[i].input_power,
                                    output_size=opt[i].output_size, 
                                    output_resolution=opt[i].output_res,
                                    fabrication=fab_function, 
                                    refraction_index=opt[i].refraction_index,
                                    normalize=opt[i].normalize,
                                    writer=writer, 
                                    device=device)
            elif opt[i].method == 'BS':
                opt_algorithm = BS(opt[i].prop_dist, opt[i].wavelength, 
                                    opt[i].feature_size, 
                                    opt[i].optim_params,
                                    opt[i].roi_res,
                                    opt[i].save_checkpoint_file,
                                    opt[i].save_wrap_res,
                                    opt[i].save_pixel_multiply, 
                                    phase_path=opt[i].root_path,
                                    prop_model=opt[i].prop_model, 
                                    propagator=propagator, 
                                    aperture_type=opt[i].aperture_type,
                                    radial_symmetry=opt[i].radial_symmetry,
                                    input_power=opt[i].input_power,
                                    slm_resolution=opt[i].slm_res,
                                    output_size=opt[i].output_size, 
                                    output_resolution=opt[i].output_res,
                                    fabrication=fab_function, 
                                    min_dose=opt[i].min_dose, 
                                    max_dos=opt[i].max_dose,
                                    refraction_index=opt[i].refraction_index,
                                    normalize=opt[i].normalize,
                                    loss_params=opt[i].loss_params, 
                                    writer=writer, 
                                    device=device)
            # upload target to GPU
            target_amp = target_amp_in[i].to(device)
            opt_algorithm.init_scale = s0
            # run algorithm (See algorithm_modules.py and algorithms.py)
            opt_algorithm.phase_path = os.path.join(opt[0].root_path)
            final_dose_tmp,final_height_tmp,_,loss_value_tmp = opt_algorithm(target_amp, init_value)
            
            if loss_value_tmp<loss_value:
                final_dose = final_dose_tmp
                loss_value = loss_value_tmp
            elif loss_value<max_loss:
                final_dose = final_dose_tmp
                break

        print('Shape: ',final_dose.shape)
        #Eval final result
        final_dose_clamped = torch.clamp(final_dose,0,255)
        final_dose_8bit = final_dose_clamped.to(torch.uint8)
        if opt[i].input_power>0.0:
            phase_shift = utils.spherical_phase(final_dose_8bit.shape, 
                                                opt[i].feature_size, 
                                                opt[i].wavelength, 
                                                1.0/opt[i].input_power)
            phase_shift = torch.tensor(phase_shift).to(device)
        else:
            phase_shift = None
        
        recon_amp_8bit = utils.forward_model(final_dose_8bit.to(final_dose.dtype), 
                                         opt[i], 
                                         propagator=propagator, 
                                         zfft2 = opt_algorithm.zfft2,
                                         fab_model=fab_function, 
                                         input_phase=phase_shift).abs().cpu().detach()
        # recon_height = utils.propagate_field(final_dose_8bit.to(final_dose.dtype), 
        #                             None, 
        #                             np.array(opt[i].prop_dist)[None,:,None,None], 
        #                             np.array(opt[i].wavelength)[None,:,None,None], 
        #                             opt[i].feature_size,
        #                             'None', 
        #                             zfft2=opt_algorithm.zfft2,
        #                             output_size=opt[i].output_size, 
        #                             output_resolution=opt[i].output_res,
        #                             fab_model=fab_function,
        #                             refraction_index=np.array(opt[i].refraction_index)[None,:,None,None],
        #                             normalize=False).abs().cpu().detach()
        recon_height = final_height_tmp.abs().cpu().detach()
        if output:
            height_out.append(recon_height)
            recon_amp.append(recon_amp_8bit)
        if plot_result:
            # prepare figure
            results = [final_dose_8bit]
            labels=['Dose']
            utils.plot_result(results,labels,normalize=False)
            results = [recon_height]
            labels=['Height']
            utils.plot_result(results,labels,normalize=False,type='plot')
            results = [recon_amp_8bit**2,target_amp**2]
            labels=['Recon','Target']
            utils.plot_result(results,labels,normalize=True,log_scale=plot_log_scale,db_min=-50)

        # gen 8-bit final result
        final_dose_8bit = final_dose_8bit.squeeze().cpu().detach().numpy()
        final_dose_8bit[final_dose_8bit>255] = 255
        final_dose_8bit[final_dose_8bit<0] = 0
        final_dose_8bit = final_dose_8bit.astype(np.uint8)
        if final_dose_8bit.ndim ==1:
            final_dose_8bit = final_dose_8bit[:,None]

        if output:
            dose_out.append(final_dose_8bit)
        # save BMP
        if opt[i].save_wrap_res is not None:
            dose_save = utils.pad_image(final_dose_8bit,
                                opt[i].save_wrap_res,
                                pytorch=False,
                                mode='wrap')
        else:
            dose_save = final_dose_8bit
        if opt[i].save_pixel_multiply is not None:
            # change pixel size
            dose_save = np.kron(dose_save,np.ones((opt[i].save_pixel_multiply,opt[i].save_pixel_multiply)))
        if batch_num >1:
            filename = os.path.join(opt[0].root_path, opt[0].run_id+'_'+str(i)+'.bmp')
        else:
            filename = os.path.join(opt[0].root_path, opt[0].run_id+'.bmp')
        cv2.imwrite(filename, dose_save.astype(np.uint8))
        # save opt
        filename = os.path.join(opt[0].root_path, opt[0].run_id+'.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(opt, f)
        print(f'    - Done, Save: ',filename)
    
        
    return dose_out, height_out ,recon_amp


def fab_opt(opt,target_height_in,
             propagator, fab_function,
             start_point='random',
             plot_result = True, plot_log_scale=True,
             output = True,batch_idx=None,
             max_loss= float("inf"), max_loop=1,
             device = torch.device('cuda')):
    '''
    Optimize dose distribution (OPE correction) 
    for target height

    Input
    ------
    opt: list
    target_phase: list
    propagator, fab_function

    Output
    ------
    dose_out
    final_recon_amp
    '''

    # Tensorboard writer
    utils.cond_mkdir(opt[0].sum_path)
    writer = SummaryWriter(opt[0].sum_path)

    # result dir
    utils.cond_mkdir('./checkpoints')
    utils.cond_mkdir(opt[0].root_path)

    # result numpy array
    batch_num = len(opt)
    dose_out = []
    height_out = []
    recon_amp = []

    # optimize for each batch
    if batch_idx is None:
        batch_idx = range(batch_num)
    for i in batch_idx:
        print('Batch # ', i)
        loss_value = float('inf')
        for j in range(max_loop):
            # upload target to GPU
            target_height = target_height_in[i].to(device)
            # use the same size as target (for fab optimization)
            slm_res = list(target_height.shape)[-2:]
            # init checkpoint/start point
            if opt[i].load_checkpoint_file is not None:
                print('Loading Checkpoints: ', opt[i].load_checkpoint_file)
                init_value,_ = utils.load_checkpoints(opt[i].load_checkpoint_file,
                                                        device)
            else:
                if start_point=='random':
                    init_value = torch.clamp(opt[i].min_dose + (opt[i].max_dose-opt[i].min_dose) * torch.rand(1, 1, *slm_res),0.0, 255.0).to(device)
                elif start_point=='inverse':
                    # ideal phase -> dose
                    init_value = fab_function(target_height,backward=True)
                    init_value = torch.clamp(init_value, opt[i].min_dose, opt[i].max_dose).to(device)
            init_value = torch.round(init_value)
            s0 = torch.tensor(1.0,device=device)
            # Select Phase generation method, algorithm
            if opt[i].method == 'SGD':
                opt_algorithm = SGD(opt[i].prop_dist, 
                                    opt[i].wavelength, 
                                    opt[i].feature_size,
                                    roi_res=slm_res, 
                                    checkpoint_filename=opt[i].save_checkpoint_file,
                                    save_wrap_res=opt[i].save_wrap_res,
                                    save_pixel_multiply=opt[i].save_pixel_multiply,
                                    phase_path=opt[i].root_path, 
                                    prop_model='None', 
                                    propagator=propagator,
                                    aperture_type = opt[i].aperture_type,
                                    input_power=0.0,
                                    slm_resolution=opt[i].slm_res,
                                    output_size=opt[i].output_size, 
                                    output_resolution=slm_res,
                                    fabrication=fab_function, 
                                    refraction_index=opt[i].refraction_index, 
                                    min_dose=opt[i].min_dose, max_dose=opt[i].max_dose,
                                    fab_var=opt[i].fab_var,
                                    normalize=opt[i].normalize,
                                    loss_params=opt[i].loss_params, 
                                    optim_params=opt[i].optim_params,
                                    s0=s0, writer=writer, device=device)
            elif opt[i].method == 'GS':
                opt_algorithm = GS(opt[i].prop_dist, 
                                    opt[i].wavelength, 
                                    opt[i].feature_size,
                                    opt[i].optim_params, 
                                    roi_res=slm_res,
                                    phase_path=opt[i].root_path, 
                                    prop_model='None', 
                                    propagator=propagator,
                                    aperture_type=opt[i].aperture_type,
                                    input_power=0.0,
                                    output_size=opt[i].output_size, 
                                    output_resolution=slm_res,
                                    fabrication=fab_function, 
                                    refraction_index=opt[i].refraction_index,
                                    normalize=opt[i].normalize,
                                    writer=writer, device=device)
            elif opt[i].method == 'BS':
                opt_algorithm = BS(opt[i].prop_dist, opt[i].wavelength, 
                                    opt[i].feature_size, 
                                    opt[i].optim_params,
                                    slm_res,
                                    opt[i].save_checkpoint_file,
                                    opt[i].save_wrap_res,
                                    opt[i].save_pixel_multiply, 
                                    opt[i].root_path,
                                    prop_model='None', 
                                    propagator=propagator,
                                    aperture_type=opt[i].aperture_type,
                                    input_power=0.0, 
                                    slm_resolution=opt[i].slm_res,
                                    output_size=opt[i].output_size, 
                                    output_resolution=slm_res,
                                    fabrication=fab_function, 
                                    min_dose=opt[i].min_dose, 
                                    max_dose=opt[i].max_dose,
                                    refraction_index=opt[i].refraction_index,
                                    normalize=opt[i].normalize,
                                    loss_params=opt[i].loss_params,
                                    writer=writer, 
                                    device=device)
               
            # run algorithm (See algorithm_modules.py and algorithms.py)
            opt_algorithm.phase_path = os.path.join(opt[0].root_path)
            final_dose_tmp,final_height_tmp,_,loss_value_tmp = opt_algorithm(target_height, init_value)
            
            if loss_value_tmp<loss_value:
                final_dose = final_dose_tmp
                loss_value = loss_value_tmp
            elif loss_value<max_loss:
                final_dose = final_dose_tmp
                break

        print('Shape: ',final_dose.shape)
        #Eval final result
        final_dose_clamped = torch.clamp(final_dose,0,255)
        final_dose_8bit = final_dose_clamped.to(torch.uint8)
        recon_amp_8bit = utils.forward_model(final_dose_8bit.to(final_dose.dtype), 
                                         opt[i], 
                                         propagator=propagator, 
                                         zfft2 = opt_algorithm.zfft2,
                                         fab_model=fab_function).abs().cpu().detach()
        # recon_height = utils.propagate_field(final_dose_8bit.to(final_dose.dtype), 
        #                             None, 
        #                             np.array(opt[i].prop_dist)[None,:,None,None], 
        #                             np.array(opt[i].wavelength)[None,:,None,None], 
        #                             opt[i].feature_size,
        #                             'None', 
        #                             zfft2=opt_algorithm.zfft2,
        #                             output_size=opt[i].output_size, 
        #                             output_resolution=opt[i].output_res,
        #                             fab_model=fab_function,
        #                             refraction_index=np.array(opt[i].refraction_index)[None,:,None,None],
        #                             normalize=False).abs().cpu().detach()
        recon_height = final_height_tmp.abs().cpu().detach()
        if output:
            height_out.append(recon_height)
            recon_amp.append(recon_amp_8bit)
        if plot_result:
            # prepare figure
            results = [final_dose_8bit]
            labels=['Dose']
            utils.plot_result(results,labels,normalize=False)
            results = [recon_height,target_height]
            labels=['Recon Height','Target Height']
            utils.plot_result(results,labels,normalize=False,type='plot')
            results = [recon_amp_8bit**2]
            labels=['Recon Intensity']
            utils.plot_result(results,labels,normalize=True,log_scale=plot_log_scale,db_min=-50)
        opt_algorithm.init_scale = torch.tensor(1.0,device=device)  
        # gen 8-bit final result
        final_dose_8bit = final_dose_8bit.squeeze().cpu().detach().numpy()
        final_dose_8bit[final_dose_8bit>255] = 255
        final_dose_8bit[final_dose_8bit<0] = 0
        final_dose_8bit = final_dose_8bit.astype(np.uint8)
        if final_dose_8bit.ndim ==1:
            final_dose_8bit = final_dose_8bit[:,None]
            
        if output:
            dose_out.append(final_dose_8bit)
        # save BMP
        if opt[i].save_wrap_res is not None:
            dose_save = utils.pad_image(final_dose_8bit,
                                opt[i].save_wrap_res,
                                pytorch=False,
                                mode='wrap')
        else:
            dose_save = final_dose_8bit
        if opt[i].save_pixel_multiply is not None:
            # change pixel size
            dose_save = np.kron(dose_save,np.ones((opt[i].save_pixel_multiply,opt[i].save_pixel_multiply)))
        if batch_num >1:
            filename = os.path.join(opt[0].root_path, opt[0].run_id+'_'+str(i)+'.bmp')
        else:
            filename = os.path.join(opt[0].root_path, opt[0].run_id+'.bmp')
        cv2.imwrite(filename, dose_save.astype(np.uint8))
        print(f'    - Done, Save: ',filename)
    return dose_out, height_out ,recon_amp



def height_opt(opt,target_amp_in,
             propagator,height_2pi,
             s0=1.0,
             plot_result = True, plot_log_scale=True,
             output = True,batch_idx=None,
             max_loss= float("inf"), max_loop=1,
             device = torch.device('cuda')):
    '''
    Optimize height map for target amp

    Input
    ------
    opt: list
    target_amp: list
    propagator, fab_function

    Output
    ------
    height_out
    final_recon_amp
    '''

    s0 =  torch.tensor(s0,device=device)
    # result numpy array
    batch_num = len(opt)
    height_out = []
    recon_amp = []

    # optimize for each batch
    if batch_idx is None:
        batch_idx = range(batch_num)
    for i in batch_idx:
        print('Batch # ', i)
        loss_value = float('inf')
        for j in range(max_loop):
            init_value = (height_2pi * torch.rand(1, 1, *opt[i].slm_res)).to(device)
            # Select Phase generation method, algorithm
            if opt[i].method_phase == 'SGD':
                opt_algorithm = SGD(opt[i].prop_dist, opt[i].wavelength, 
                                    opt[i].feature_size_phase,
                                    opt[i].roi_res, 
                                    opt[i].save_checkpoint_file,
                                    opt[i].save_wrap_res,
                                    opt[i].save_pixel_multiply,
                                    phase_path=opt[i].root_path, 
                                    prop_model=opt[i].prop_model, 
                                    propagator=propagator,
                                    aperture_type = opt[i].aperture_type,
                                    radial_symmetry=opt[i].radial_symmetry,
                                    input_power=opt[i].input_power,
                                    slm_resolution=opt[i].slm_res,
                                    output_size=opt[i].output_size, 
                                    output_resolution=opt[i].output_res,
                                    fabrication=None, 
                                    refraction_index=opt[i].refraction_index, 
                                    min_dose=0.0,max_dose=height_2pi,
                                    fab_var=opt[i].fab_var,
                                    normalize=opt[i].normalize,
                                    loss_params=opt[i].loss_params_phase, 
                                    optim_params=opt[i].optim_params_phase,
                                    s0=s0,
                                    writer=None, device=device)
            elif opt[i].method_phase == 'GS':
                opt_algorithm = GS(opt[i].prop_dist, opt[i].wavelength, 
                                    opt[i].feature_size_phase, 
                                    opt[i].optim_params_phase,
                                    roi_res=opt[i].roi_res,
                                    phase_path=opt[i].root_path, 
                                    prop_model=opt[i].prop_model, 
                                    propagator=propagator,
                                    aperture_type=opt[i].aperture_type,
                                    input_power=opt[i].input_power,
                                    output_size=opt[i].output_size, 
                                    output_resolution=opt[i].output_res,
                                    fabrication=None, 
                                    refraction_index=opt[i].refraction_index,
                                    normalize=opt[i].normalize,
                                    writer=None, 
                                    device=device)
            elif opt[i].method_phase == 'BS':
                opt_algorithm = BS(opt[i].prop_dist, 
                                    opt[i].wavelength, 
                                    opt[i].feature_size_phase, 
                                    opt[i].optim_params_phase,
                                    opt[i].roi_res,
                                    checkpoint_filename=opt[i].save_checkpoint_file, 
                                    save_wrap_res=opt[i].save_wrap_res,
                                    save_pixel_multiply=opt[i].save_pixel_multiply,
                                    phase_path=opt[i].root_path,
                                    prop_model=opt[i].prop_model, 
                                    propagator=propagator, 
                                    aperture_type=opt[i].aperture_type, 
                                    input_power=opt[i].input_power,
                                    slm_resolution=opt[i].slm_res,
                                    output_size=opt[i].output_size, 
                                    output_resolution=opt[i].output_res,
                                    fabrication=None, 
                                    max_dose=height_2pi,
                                    refraction_index=opt[i].refraction_index,
                                    normalize=opt[i].normalize,
                                    loss_params=opt[i].loss_params_phase, 
                                    writer=None, 
                                    device=device)
            # upload target to GPU
            target_amp = target_amp_in[i].to(device)
            opt_algorithm.init_scale = s0
            # run algorithm (See algorithm_modules.py and algorithms.py)
            opt_algorithm.phase_path = os.path.join(opt[0].root_path)
            final_height_tmp,_,_,loss_value_tmp = opt_algorithm(target_amp, init_value)
            
            if loss_value_tmp<loss_value:
                final_height = final_height_tmp
                loss_value = loss_value_tmp
            elif loss_value<max_loss:
                final_height = final_height_tmp
                break

        print('Shape: ',final_height.shape)
        #Eval final result
        final_height = torch.clamp(final_height,0.0,height_2pi)
        if opt[i].input_power>0.0:
            phase_shift = utils.spherical_phase(final_height.shape, 
                                                opt[i].feature_size_phase, 
                                                opt[i].wavelength, 
                                                1.0/opt[i].input_power)
            phase_shift = torch.tensor(phase_shift).to(device)
        else:
            phase_shift = None

        recon_amp_tmp = utils.forward_model(final_height, 
                                         opt[i], 
                                         propagator=propagator, 
                                         zfft2 = opt_algorithm.zfft2,
                                         input_phase=phase_shift).abs().cpu().detach()
        if output:
            recon_amp.append(recon_amp_tmp)
        if plot_result:
            # prepare figure
            results = [final_height]
            labels=['Height']
            utils.plot_result(results,labels,normalize=False)
            results = [recon_amp_tmp**2,target_amp**2]
            labels=['Recon','Target']
            utils.plot_result(results,labels,normalize=True,log_scale=plot_log_scale,db_min=-50)

        # final result
        # final_height = final_height.cpu().detach()
        final_height = final_height.detach()
        # change pixel size
        kernel_size = [int(round(s1/s2)) for s1,s2 in zip(opt[i].feature_size_phase,opt[i].feature_size)]
        pixel_kernel = torch.ones(kernel_size,device=final_height.device)
        final_height = torch.kron(final_height,pixel_kernel)
        if output:
            height_out.append(final_height)
        
        print(f'    - Phase retrieval Done')
        # save file
        filename = os.path.join(opt[0].root_path,opt[0].run_id+'_phase.pt')
        torch.save(final_height, filename)
        print(f'    - Done, Save: ',filename)
        # save opt
        filename = os.path.join(opt[0].root_path, opt[0].run_id+'.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(opt, f)
    return height_out ,recon_amp
