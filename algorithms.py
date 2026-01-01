import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
import numpy as np

import utils.utils as utils

from physical_model import *


# 1. GS
def gerchberg_saxton(init_value, target_amp, 
                     optim_params, prop_dist, 
                     wavelength, feature_size=6.4e-6,
                     phase_path=None, prop_model='ASM', propagator=None,
                     input_phase = None,
                     aperture_type='square',
                     zfft2=None,
                     output_size=None,output_resolution=None,
                     roi_res=None, 
                     fab_model=None, 
                     refraction_index=None,
                     normalize=True,
                     writer=None, dtype=torch.float32, precomputed_H_f=None, precomputed_H_b=None):
    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator

    :param init_phase: a tensor, in the shape of (1,1,H,W), initial guess for the phase.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param phase_path: path to save the results.
    :param prop_model: string indicating the light transport model, default 'ASM'. ex) 'ASM', 'fresnel', 'model'
    :param propagator: predefined function or model instance for the propagation.
    :param writer: tensorboard writer
    :param dtype: torch datatype for computation at different precision, default torch.float32.
    :param precomputed_H_f: A Pytorch complex64 tensor, pre-computed kernel for forward prop (SLM to image)
    :param precomputed_H_b: A Pytorch complex64 tensor, pre-computed kernel for backward propagation (image to SLM)

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """
    # write tensorboard every show_iters steps
    show_iters = 100
    device = init_value.device
    # loss function (used for eval.)
    loss=nn.MSELoss()
    x = init_value
    if aperture_type == 'round':
        x = utils.round_masked(x)

    # padding target
    target_amp = utils.crop_pad_image(target_amp,output_resolution)
    roi_mask = utils.crop_mask(target_amp,roi_res)
  
    # run the GS algorithm
    for k in range(optim_params.num_iters):
        # SLM plane to image plane
        # w/fab_model, dose->height
        # w/o fab_model, height->height
        recon_height = utils.fabrication_field(x, 
                        aperture_type=aperture_type,
                        fab_model=fab_model, 
                        fab_backward=False)
        # w/prop_model, height->complex field
        # w/o prop_model, no change (still height)
        recon_field = utils.propagate_field(recon_height, propagator, 
                        prop_dist=prop_dist, 
                        wavelength=wavelength, 
                        refraction_index=refraction_index,
                        feature_size=feature_size,
                        prop_model=prop_model, 
                        prop_backward=False,
                        input_phase=input_phase,
                        aperture_type=aperture_type,
                        zfft2=zfft2,
                        output_size=output_size, 
                        output_resolution=output_resolution,
                        normalize=normalize,
                        field_type='height',
                        dtype=dtype,
                        precomputed_H=precomputed_H_f)
        
        # write to tensorboard / write phase image
        # Note that it takes 0.~ s for writing it to tensorboard 
        if k > 0 and k % show_iters == 0:
            lossValue = loss(recon_field.abs()[roi_mask], target_amp[roi_mask])
            print(k)
            print('Loss: ',lossValue.cpu().detach().numpy())
            utils.write_sgd_summary(x, recon_field.abs()[0,:,:,:].unsqueeze(0), 
                                    target_amp[0,:,:,:].unsqueeze(0),
                                    lossValue.cpu().detach().numpy(), 
                                    k,
                                    writer=writer, 
                                    path=phase_path, prefix='gs')
        
        if prop_model != 'None':
            # for height (phase) optimization & e2e optimization
            # replace amplitude at the image plane
            recon_field[roi_mask] = torch.exp(1j*recon_field[roi_mask].angle()) * target_amp[roi_mask]
        else:
            # for fabrication optimization
            # use Hybrid Input/Output strategy
            recon_field = (recon_field+target_amp)/2
        
        # image plane to SLM phase (reverse prop.)
        # w/prop_model, complex field -> height
        # w/o prop_model, no change (still height)
        recon_height = utils.propagate_field(recon_field, 
                        propagator, 
                        prop_dist=-prop_dist, 
                        wavelength=wavelength, 
                        refraction_index=refraction_index,
                        feature_size=feature_size,
                        prop_model=prop_model,
                        prop_backward=True, 
                        input_phase=None,
                        aperture_type=aperture_type,
                        zfft2=zfft2,
                        output_size=output_size, 
                        output_resolution=output_resolution,
                        normalize=normalize,
                        field_type='comp',
                        dtype=dtype,
                        precomputed_H=precomputed_H_b)
        # backward fab model 
        # w/ fab_model: height->dose
        # w/o fab_model: height->height (no change)
        x = utils.fabrication_field(recon_height, 
                        aperture_type,
                        fab_model, 
                        fab_backward=True)
        
    # plot
    # utils.plot_result([x.angle(),recon_field.abs(),output_dose],
    #                    labels=['phase','recon','dose'])
    # return phases,amps
    if aperture_type == 'round':
        x = utils.round_masked(x)
    return x,x,recon_field.abs(),lossValue.cpu().detach().numpy()


# 2. SGD
def stochastic_gradient_descent(init_value, target_amp, 
                                prop_dist, wavelength, 
                                feature_size,
                                checkpoint_filename,
                                phase_path=None, 
                                prop_model='ASM', propagator=None,
                                input_phase = None,
                                slm_resolution=None,
                                aperture_type='square',
                                radial_symmetry=False,
                                zfft2=None,
                                output_size=None,output_resolution=None,
                                fab_model=None, fab_var=None,
                                refraction_index=None,
                                min_dose=0.0,max_dose=255.0,
                                roi_res=None,
                                normalize=True,
                                loss_params=None, 
                                optim_params=None, 
                                s0=1.0, 
                                writer=None, 
                                save_wrap_res=None,
                                save_pixel_multiply=None, dtype=torch.float32, precomputed_H=None):
    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator.

    Input
    ------
    :param init_value: a tensor, in the shape of (1,1,H,W), initial guess.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the SGD.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param refraction_index: refraction index, needed by fab_model to get phase
    :param roi_res: a tuple of integer, region of interest, like (880, 1600)
    :param phase_path: a string, for saving intermediate phases
    :param prop_model: a string, that indicates the propagation model. ('ASM' or 'MODEL')
    :param propagator: predefined function or model instance for the propagation.
    :param loss: loss function, default L2
    :param lr: learning rate for optimization variables
    :param lr_s: learning rate for learnable scale
    :param s0: initial scale
    :param writer: Tensorboard writer instance
    :param dtype: default torch.float32
    :param precomputed_H: A Pytorch complex64 tensor, pre-computed kernel shape of (1,1,2H,2W) for fast computation.

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """
    
    # write tensorboard every show_iters steps
    show_iters = 100

    device = init_value.device
    s = s0.clone().detach().to(device=device).requires_grad_(True)
    min_dose = torch.tensor(min_dose, device=device)
    max_dose = torch.tensor(max_dose, device=device)

    x = init_value.requires_grad_(True)
    
    # optimization variables and adam optimizer
    optvars = [{'params': x}]
    if optim_params.lr_s > 0:
        optvars += [{'params': s, 'lr': optim_params.lr_s}]
    
    if optim_params.optimizer=='adam':
        optimizer = optim.Adam(optvars, 
                                lr=optim_params.lr, 
                                amsgrad=optim_params.amsgrad,
                                eps=optim_params.eps)
    elif optim_params.optimizer=='sgd':
        optimizer = optim.SGD(optvars,lr=optim_params.lr,eps=optim_params.eps,
                            momentum=0.9, 
                            weight_decay=0.0005,
                            nesterov=True)
    # resize target to the size of roi_res
    target_amp = utils.crop_pad_image(target_amp,roi_res)
    if fab_var is not None:
        batch_num = len(fab_var['gamma'])
        target_amp = target_amp.expand(batch_num, -1,-1,-1)
    
    if optim_params.mask_grad_percentage<0.99:
        # setting some gradient value to zero
        mask = utils.mask_gradients(x,optim_params.mask_grad_percentage)
        @torch.utils.hooks.unserializable_hook
        def mask_gradients(grad):
            return grad * mask
        x.register_hook(mask_gradients)
    
    if aperture_type == 'round':
        x_sym = utils.round_masked(x_sym)
    
    
    
    # run the iterative algorithm
    for k in range(optim_params.num_iters): 
        optimizer.zero_grad()
        
        if k % 10 ==0:
            if optim_params.mask_grad_percentage<0.99:
                # update gradients' mask
                mask = utils.mask_gradients(x,optim_params.mask_grad_percentage)
        
        
        if radial_symmetry: 
            # impose radial symmetry
            # interpolate the 1D tensor to a 2D tensor with radial symmetry
            sz = [x.shape[0],x.shape[1],slm_resolution[0],slm_resolution[1]]
            x_sym = utils.radial_symmetric_interpolation(x,shape=sz) 
        else:
            x_sym = x 

        
        # forward propagation
        # shape: [B,C,H,W], B for fab_var; C for wavelengths
        # forward propagation model
        if fab_var is not None:
            recon_field = []
            recon_height = []
            for gamma,amp in zip(fab_var['gamma'],fab_var['amp']):
                fab_model.gamma = gamma
                fab_model.amp = amp
                recon_height_tmp = utils.fabrication_field(
                                        torch.clamp(x_sym,min_dose,max_dose), 
                                        aperture_type=aperture_type,
                                        fab_model=fab_model, 
                                        fab_backward=False)
                recon_height.append(recon_height_tmp)                        
                recon_field_tmp = utils.propagate_field(recon_height_tmp, 
                                        propagator, 
                                        prop_dist=prop_dist, 
                                        wavelength=wavelength, 
                                        refraction_index=refraction_index,
                                        feature_size=feature_size,
                                        prop_model=prop_model, 
                                        input_phase=input_phase,
                                        prop_backward=False,
                                        aperture_type=aperture_type,
                                        zfft2=zfft2,
                                        output_size=output_size, 
                                        output_resolution=output_resolution,
                                        normalize=normalize,
                                        field_type='height',
                                        dtype=dtype,
                                        precomputed_H=precomputed_H)
                recon_field.append(recon_field_tmp)
            recon_height = torch.cat(recon_height)
            recon_field = torch.cat(recon_field)
        else:
            recon_height = utils.fabrication_field(torch.clamp(x_sym,min_dose,max_dose), 
                                        aperture_type=aperture_type,
                                        fab_model=fab_model, 
                                        fab_backward=False)
            
            recon_field = utils.propagate_field(recon_height, 
                                        propagator, 
                                        prop_dist=prop_dist, 
                                        wavelength=wavelength, 
                                        refraction_index=refraction_index,
                                        feature_size=feature_size,
                                        prop_model=prop_model, 
                                        input_phase=input_phase,
                                        prop_backward=False,
                                        aperture_type=aperture_type,
                                        zfft2=zfft2,
                                        output_size=output_size, 
                                        output_resolution=output_resolution,
                                        normalize=normalize,
                                        field_type='height',
                                        dtype=dtype,
                                        precomputed_H=precomputed_H)

        # get amplitude
        recon_amp = recon_field.abs().to(dtype=dtype)
        # get SLM_phase
        recon_phase = utils.height2phase(recon_height,wavelength,refraction_index,dtype)
        # calculate loss and backprop
        if loss_params.spectrum_penalty_weight>0:
            cutoff_frequency = min([1.0,
                                    loss_params.spectrum_relative_fwhm*\
                                    2*max(feature_size)/\
                                    min(wavelength.ravel())*\
                                    math.sin(math.atan(max(init_value.shape)*\
                                                        max(feature_size)/2/\
                                                        max(prop_dist.ravel())))
                                    ])
            if k==1:
                print('Spectrum cutoff ratio: ',cutoff_frequency)
        else:
            cutoff_frequency = 0.5
        
        lossValue = utils.loss(recon_amp,recon_phase,target_amp,
                               loss_params,
                               cutoff_frequency=cutoff_frequency,
                               roi_res=roi_res,
                               s=s)
        lossValue.backward()
        optimizer.step()

        # write to tensorboard / write phase(dose) image
        # Note that it takes 0.~ s for writing it to tensorboard
        # print(k)
        with torch.no_grad():
            if k % show_iters == 1:
                print(k)
                lossValue = utils.loss(recon_amp,recon_phase,target_amp,
                               loss_params,
                               cutoff_frequency=cutoff_frequency,
                               roi_res=roi_res,
                               s=1.0)
                print('Loss: ',lossValue.cpu().detach().numpy())
                height = utils.fabrication_field(x_sym, 
                                        aperture_type=aperture_type,
                                        fab_model=fab_model, 
                                        fab_backward=False)
                # utils.write_sgd_summary(x_sym, recon_amp[0,:,:,:].unsqueeze(0), 
                #                         height[0,:,:,:].unsqueeze(0),
                #                         lossValue.cpu().detach().numpy(), 
                #                         k,
                #                         writer=writer, 
                #                         path=phase_path, 
                #                         s=s, prefix='sgd')
                utils.save_checkpoints(checkpoint_filename,x_sym,s)
                utils.save_tmp_bmp(x_sym,num=k,save_wrap_res=save_wrap_res,save_pixel_multiply=save_pixel_multiply)
    
    height = utils.fabrication_field(x_sym, aperture_type=aperture_type,
                                    fab_model=fab_model, 
                                    fab_backward=False)
    if aperture_type == 'round':
        x_sym = utils.round_masked(x_sym)
    return x_sym,height,recon_amp,lossValue.cpu().detach().numpy()


# 3. DBS
def binary_search(init_value, target_amp, 
                optim_params, 
                prop_dist, wavelength, feature_size,
                checkpoint_filename, phase_path=None, 
                fab_model=None, 
                min_dose=0.0,max_dose=255.0,levels=255,
                refraction_index=None, 
                prop_model='ASM', propagator=None,
                input_phase=None,
                slm_resolution=None,
                aperture_type='square',
                radial_symmetry=False,
                zfft2=None,
                output_size=None,output_resolution=None,
                roi_res=None,
                normalize=True,
                loss_params=None, 
                writer=None, 
                save_wrap_res=None,
                save_pixel_multiply=None,
                dtype=torch.float32, precomputed_H=None):
    """
    Given the initial guess, run the SGD algorithm to calculate the optimal phase pattern of spatial light modulator.

    Input
    ------
    :param init_value: a tensor, in the shape of (1,1,H,W), initial guess.
    :param target_amp: a tensor, in the shape of (1,1,H,W), the amplitude of the target image.
    :param num_iters: the number of iterations to run the SGD.
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength in m.
    :param feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    :param refraction_index: refraction index, needed by fab_model to get phase
    :param roi_res: a tuple of integer, region of interest, like (880, 1600)
    :param phase_path: a string, for saving intermediate phases
    :param prop_model: a string, that indicates the propagation model. ('ASM' or 'MODEL')
    :param propagator: predefined function or model instance for the propagation.
    :param loss: loss function, default L2
    :param lr: learning rate for optimization variables
    :param lr_s: learning rate for learnable scale
    :param s0: initial scale
    :param writer: Tensorboard writer instance
    :param dtype: default torch.float32
    :param precomputed_H: A Pytorch complex64 tensor, pre-computed kernel shape of (1,1,2H,2W) for fast computation.

    Output
    ------
    :return: a tensor, the optimized phase pattern at the SLM plane, in the shape of (1,1,H,W)
    """
    # write tensorboard every show_iters steps
    show_iters = 10

    device = init_value.device
    x = init_value.requires_grad_(False)
    if aperture_type == 'round':
        x_sym = utils.round_masked(x)
    min_dose = torch.tensor(min_dose, device=device)
    max_dose = torch.tensor(max_dose, device=device)
    delta = torch.tensor((max_dose-min_dose)/levels, device=device)
    

    # resize target
    target_amp = utils.crop_pad_image(target_amp,roi_res)

    # init
    # forward propagation, get amplitude
    if radial_symmetry: 
        # impose radial symmetry
        # interpolate the 1D tensor to a 2D tensor with radial symmetry
        sz = [x.shape[0],x.shape[1],slm_resolution[0],slm_resolution[1]]
        x_sym = utils.radial_symmetric_interpolation(x,shape=sz) 
    else:
        x_sym = x 
    recon_amp = utils.propagate_field(
                    utils.fabrication_field(x_sym,aperture_type,
                                            fab_model, 
                                            fab_backward=False), 
                    propagator, 
                    prop_dist, wavelength, refraction_index, feature_size,
                    prop_model=prop_model, prop_backward=False,
                    input_phase=input_phase,aperture_type=aperture_type,
                    zfft2=zfft2,
                    output_size=output_size, 
                    output_resolution=output_resolution,
                    normalize=normalize,
                    field_type='height',
                    dtype=dtype,
                    precomputed_H=precomputed_H).abs()
    # calculate loss 
    lossValue0 = utils.loss(recon_amp,None,target_amp,loss_params,roi_res)

    # run the iterative algorithm
    for k in range(optim_params.num_iters):
        rand_index = np.random.permutation(len(x.view(-1)))
        for i in rand_index:
            # add '+1'
            x.view(-1)[i] = utils.circular_clamp(x.view(-1)[i]+delta, min_dose, max_dose)
            if radial_symmetry: 
                # impose radial symmetry
                # interpolate the 1D tensor to a 2D tensor with radial symmetry
                sz = [x.shape[0],x.shape[1],slm_resolution[0],slm_resolution[1]]
                x_sym = utils.radial_symmetric_interpolation(x,shape=sz) 
            else:
                x_sym = x 
            # forward propagation
            recon_amp = utils.propagate_field(
                            utils.fabrication_field(x_sym,aperture_type,
                                                    fab_model, 
                                                    fab_backward=False), 
                            propagator, 
                            prop_dist, wavelength, refraction_index, feature_size,
                            prop_model=prop_model, prop_backward=False,
                            input_phase=input_phase,aperture_type=aperture_type,
                            zfft2=zfft2,
                            output_size=output_size, 
                            output_resolution=output_resolution,
                            normalize=normalize,
                            field_type='height',
                            dtype=dtype,
                            precomputed_H=precomputed_H).abs()
            # calculate loss 
            lossValue = utils.loss(recon_amp,None,target_amp,loss_params,roi_res)
            if lossValue<lossValue0:
                lossValue0 = lossValue
            else:
                # add '-1'
                x.view(-1)[i] = utils.circular_clamp(x.view(-1)[i]-2*delta, min_dose, max_dose)
                if radial_symmetry: 
                    # impose radial symmetry
                    # interpolate the 1D tensor to a 2D tensor with radial symmetry
                    sz = [x.shape[0],x.shape[1],slm_resolution[0],slm_resolution[1]]
                    x_sym = utils.radial_symmetric_interpolation(x,shape=sz) 
                else:
                    x_sym = x 
                # forward propagation
                recon_amp = utils.propagate_field(
                                utils.fabrication_field(x_sym,aperture_type,
                                                        fab_model, 
                                                        fab_backward=False), 
                                propagator, 
                                prop_dist, wavelength, refraction_index, feature_size,
                                prop_model=prop_model, prop_backward=False,
                                input_phase=input_phase,aperture_type=aperture_type,
                                zfft2=zfft2,
                                output_size=output_size, 
                                output_resolution=output_resolution,
                                normalize=normalize,
                                field_type='height',
                                dtype=dtype,
                                precomputed_H=precomputed_H).abs()
                # calculate loss 
                lossValue = utils.loss(recon_amp,None,target_amp,loss_params,roi_res)
                if lossValue>lossValue0:
                    # resume x
                    x.view(-1)[i] = utils.circular_clamp(x.view(-1)[i]+delta, min_dose, max_dose)
                    if radial_symmetry: 
                        # impose radial symmetry
                        # interpolate the 1D tensor to a 2D tensor with radial symmetry
                        sz = [x.shape[0],x.shape[1],slm_resolution[0],slm_resolution[1]]
                        x_sym = utils.radial_symmetric_interpolation(x,shape=sz) 
                    else:
                        x_sym = x 

        # write to tensorboard / write phase(dose) image
        # Note that it takes 0.~ s for writing it to tensorboard
        
        if k % show_iters == 1:
            print(k)
            print('Loss: ',lossValue.cpu().numpy())
            utils.write_sgd_summary(x_sym, recon_amp[0,:,:,:].unsqueeze(0), 
                                    target_amp[0,:,:,:].unsqueeze(0),
                                    lossValue0.sqrt(), k,
                                    writer=writer, path=phase_path, prefix='bs')
            s = torch.tensor(1.0,dtype=dtype,device=device)
            utils.save_checkpoints(checkpoint_filename,x,s)
    height = utils.fabrication_field(x_sym,aperture_type,
                                     fab_model, 
                                     fab_backward=False)
    if aperture_type == 'round':
        x_sym = utils.round_masked(x)
    return x_sym,height,recon_amp,lossValue.cpu().numpy()
