"""
Some modules for easy use. (No need to calculate kernels explicitly)

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms import gerchberg_saxton, stochastic_gradient_descent, binary_search

import os
import time
import skimage.io
from physical_model import fabrication_model
import utils.utils as utils
import platform
my_os = platform.system()

class LaplacianLoss(nn.Module):
    def __init__(self, device):
        super(LaplacianLoss, self).__init__()
        self.kernel = self.laplacian_kernel(device)
        
    @staticmethod
    def laplacian_kernel(device):
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        return kernel.to(device)
    
    def forward(self, input_image, target_image):
        input_laplacian = F.conv2d(input_image, self.kernel, padding=1)
        target_laplacian = F.conv2d(target_image, self.kernel, padding=1)
        loss = F.mse_loss(input_laplacian, target_laplacian)
        return loss

class GS(nn.Module):
    """Classical Gerchberg-Saxton algorithm

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param num_iters: the number of iteration, default 500
    :param phase_path: path to write intermediate results
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> gs = GS(...)
    >>> final_phase = gs(target_amp, init_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    init_phase: initial guess of phase of phase-only slm
    final_phase: optimized phase-only representation at SLM plane, same dimensions
    """
    def __init__(self, prop_dist, wavelength, 
                 feature_size, optim_params,
                 roi_res = None,
                 phase_path=None,
                 prop_model='ASM', propagator=None, 
                 aperture_type = 'square',
                 input_power=0.0,
                 output_size = None, output_resolution=None,
                 fabrication=None,refraction_index=None,
                 normalize=True,
                 writer=None, device=torch.device('cuda'),dtype=torch.float32):
        super(GS, self).__init__()

        # Setting parameters
        self.prop_dist = np.array(prop_dist[0])
        self.wavelength = np.array(wavelength[0])
        self.feature_size = feature_size
        self.phase_path = phase_path
        self.precomputed_H_f = None # propagation forward kernel
        self.precomputed_H_b = None # propagation backward kernel
        self.prop_model = prop_model
        self.prop = propagator
        self.input_power=input_power
        self.input_phase = None
        self.aperture_type = aperture_type
        self.zfft2 = None
        self.output_size = output_size
        self.output_resolution=output_resolution
        self.fab = fabrication
        self.refraction_index = np.array(refraction_index[0])
        self.optim_params = optim_params
        self.roi_res = roi_res
        self.writer = writer
        self.dev = device
        self.normalize=normalize
        self.dtype=dtype
        self.init_scale = 1.0

    def forward(self, target_amp, init_phase=None):
        # Pre-compute propagataion kernel only once
        if self.precomputed_H_f is None:
            if self.prop_model == 'ASM':
                self.precomputed_H_f = self.prop(torch.empty(*init_phase.shape, dtype=torch.complex64), self.feature_size,
                                                self.wavelength, self.prop_dist,
                                                self.output_resolution,
                                                return_H=True)
                self.precomputed_H_f = self.precomputed_H_f.to(self.dev).detach()
                self.precomputed_H_f.requires_grad = False
            if self.prop_model == 'SFR':
                self.zfft2 = self.prop(torch.empty(*init_phase.shape, dtype=torch.complex64, device=self.dev), self.feature_size,
                                                self.wavelength, self.prop_dist, 
                                                self.output_size,self.output_resolution,
                                                return_zfft2=True, dtype=self.dtype)
                
                self.precomputed_H_f = self.prop(torch.empty(*init_phase.shape, dtype=torch.complex64), self.feature_size,
                                                self.wavelength, self.prop_dist, 
                                                self.output_size,self.output_resolution,
                                                return_H=True, dtype=self.dtype)
                self.precomputed_H_f = [H.to(self.dev).detach() for H in self.precomputed_H_f]
                for H in self.precomputed_H_f:
                    H.requires_grad = False
        if self.precomputed_H_b is None:
            if self.prop_model == 'ASM':
                self.precomputed_H_b = self.prop(torch.empty(*init_phase.shape, dtype=torch.complex64), self.feature_size,
                                                self.wavelength, -self.prop_dist, 
                                                self.output_resolution,
                                                return_H=True)
                self.precomputed_H_b = self.precomputed_H_b.to(self.dev).detach()
                self.precomputed_H_b.requires_grad = False
            if self.prop_model == 'SFR':
                self.zfft2 = self.prop(torch.empty(*init_phase.shape, dtype=torch.complex64, device=self.dev), self.feature_size,
                                                self.wavelength, self.prop_dist, 
                                                self.output_size,self.output_resolution,
                                                return_zfft2=True, dtype=self.dtype)
                
                
                self.precomputed_H_b = self.prop(torch.empty(*init_phase.shape, dtype=torch.complex64), self.feature_size,
                                                self.wavelength, -self.prop_dist, 
                                                self.output_size,self.output_resolution,
                                                return_H=True, dtype=self.dtype)
                self.precomputed_H_b = [H.to(self.dev).detach() for H in self.precomputed_H_b]
                for H in self.precomputed_H_b:
                    H.requires_grad = False
        # Pre-compute fabrication kernel only once
        # if self.precomputed_F_f is None:


        if self.input_power>0 and self.input_phase is None:
            phase_shift = utils.spherical_phase(init_phase.shape, 
                                                self.feature_size, 
                                                self.wavelength, 
                                                1.0/self.input_power)
            self.input_phase = torch.tensor(phase_shift,dtype=self.dtype).to(self.dev).detach()
        # Run algorithm
        final_dose,final_height,final_amp,loss_value = gerchberg_saxton(init_phase, target_amp, self.optim_params, self.prop_dist,
                                        self.wavelength, self.feature_size,
                                        phase_path=self.phase_path,
                                        prop_model=self.prop_model, propagator=self.prop,
                                        input_phase=self.input_phase,
                                        aperture_type=self.aperture_type,
                                        zfft2=self.zfft2,
                                        output_size=self.output_size, output_resolution=self.output_resolution,
                                        precomputed_H_f=self.precomputed_H_f, precomputed_H_b=self.precomputed_H_b,
                                        roi_res = self.roi_res,
                                        fab_model=self.fab,
                                        refraction_index=self.refraction_index,
                                        normalize=self.normalize,
                                        writer=self.writer)
        return final_dose,final_height,final_amp,loss_value

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path

class SGD(nn.Module):
    """Proposed Stochastic Gradient Descent Algorithm using Auto-diff Function of PyTorch

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param roi_res: region of interest to penalize the loss
    :param phase_path: path to write intermediate results
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param lr: learning rate for phase variables
    :param lr_s: learning rate for the learnable scale
    :param s0: initial scale
    :param writer: SummaryWrite instance for tensorboard
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> sgd = SGD(...)
    >>> final_phase = sgd(target_amp, init_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    init_phase: initial guess of phase of phase-only slm
    final_phase: optimized phase-only representation at SLM plane, same dimensions
    """
    def __init__(self, prop_dist, wavelength,
                 feature_size, 
                 roi_res, 
                 checkpoint_filename,
                 save_wrap_res,
                 save_pixel_multiply,
                 phase_path=None, prop_model='ASM',
                 propagator=None,
                 aperture_type='square',
                 radial_symmetry=False,
                 input_power=0.0,
                 slm_resolution=None,
                 output_size=None, output_resolution=None,
                 fabrication=None, refraction_index=None,
                 min_dose=0.0, max_dose=255.0,
                 fab_var=None,
                 normalize=True,
                 loss_params=None, 
                 optim_params=None,
                 s0=1.0,
                 writer=None, device=torch.device('cuda'), dtype=torch.float32):
        super(SGD, self).__init__()
        # Setting parameters
        # convert to tensors with shape of [1,C,1,1]
        self.fab = fabrication
        self.refraction_index = refraction_index
        self.wavelength = wavelength    
        self.prop_dist = prop_dist
        
        # convert to tensors with shape of [B,1,1,1]
        self.fab_var = fab_var
        self.min_dose = min_dose
        self.max_dose = max_dose
        self.output_size = output_size
        self.feature_size = feature_size
        self.roi_res = roi_res
        self.phase_path = phase_path
        self.precomputed_H = None
        self.prop_model = prop_model
        self.prop = propagator
        self.aperture_type = aperture_type
        self.radial_symmetry= radial_symmetry
        self.input_power=input_power
        self.slm_resolution = slm_resolution
        self.input_phase=None
        self.init_phase = None
        self.zfft2 = None
        self.output_resolution=output_resolution
        self.filename = checkpoint_filename
        self.save_wrap_res = save_wrap_res
        self.save_pixel_multiply = save_pixel_multiply

        self.normalize=normalize
        self.loss_params = loss_params
        self.optim_params = optim_params
        self.init_scale = s0

        self.writer = writer
        self.dev = device
        self.dtype = dtype

    def forward(self, target_amp, init_value=None):
        # Pre-compute propagataion kernel only once
        if self.precomputed_H is None:
            sz = [init_value.shape[0],init_value.shape[1],self.slm_resolution[0],self.slm_resolution[1]]
            if self.prop_model == 'ASM':
                self.precomputed_H = self.prop(torch.empty(*sz, dtype=torch.complex64, device=self.dev), self.feature_size,
                                                self.wavelength, self.prop_dist, 
                                                self.output_resolution,
                                                return_H=True, dtype=self.dtype)
                self.precomputed_H = self.precomputed_H.to(self.dev).detach()
                self.precomputed_H.requires_grad = False
            if self.prop_model == 'SFR':
                self.zfft2 = self.prop(torch.empty(*sz, dtype=torch.complex64, device=self.dev), self.feature_size,
                                                self.wavelength, self.prop_dist, 
                                                self.output_size,self.output_resolution,
                                                return_zfft2=True, dtype=self.dtype)
                self.precomputed_H = self.prop(torch.empty(*sz, dtype=torch.complex64), self.feature_size,
                                                self.wavelength, self.prop_dist, 
                                                self.output_size,self.output_resolution,
                                                return_H=True, zfft2=self.zfft2, dtype=self.dtype)
                self.precomputed_H = [H.to(self.dev).detach() for H in self.precomputed_H]
                for H in self.precomputed_H:
                    H.requires_grad = False
        
        if self.input_power>0 and self.input_phase is None:
            phase_shift = utils.spherical_phase(self.slm_resolution, 
                                                self.feature_size, 
                                                self.wavelength, 
                                                1.0/self.input_power)
            self.input_phase = torch.tensor(phase_shift,dtype=self.dtype).to(self.dev).detach()
        # Run algorithm
        final_dose,final_height,final_amp,loss_value = stochastic_gradient_descent(init_value, target_amp, self.prop_dist,
                                                  self.wavelength, 
                                                  self.feature_size,
                                                  self.filename, 
                                                  phase_path=self.phase_path,
                                                  prop_model = self.prop_model, 
                                                  propagator = self.prop,
                                                  input_phase=self.input_phase,
                                                  slm_resolution=self.slm_resolution,
                                                  aperture_type=self.aperture_type,
                                                  radial_symmetry=self.radial_symmetry,
                                                  zfft2=self.zfft2,
                                                  output_size=self.output_size,
                                                  output_resolution=self.output_resolution,
                                                  fab_model=self.fab, 
                                                  fab_var=self.fab_var,
                                                  refraction_index = self.refraction_index,
                                                  min_dose=self.min_dose, max_dose=self.max_dose,
                                                  roi_res=self.roi_res, 
                                                  normalize=self.normalize,
                                                  loss_params=self.loss_params, 
                                                  optim_params=self.optim_params,
                                                  s0=self.init_scale, 
                                                  writer=self.writer,
                                                  save_wrap_res=self.save_wrap_res,
                                                  save_pixel_multiply=self.save_pixel_multiply,
                                                  dtype=self.dtype,
                                                  precomputed_H=self.precomputed_H)
        return final_dose,final_height,final_amp,loss_value

    def zfft2(self):
        return self.zfft2

    @property
    def init_scale(self):
        return self._init_scale

    @init_scale.setter
    def init_scale(self, s):
        self._init_scale = s

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path

    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self, prop):
        self._prop = prop


class BS(nn.Module):
    """binary search

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param roi_res: region of interest to penalize the loss
    :param phase_path: path to write intermediate results
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param writer: SummaryWrite instance for tensorboard
    :param device: torch.device
    """
    def __init__(self, prop_dist, wavelength, 
                 feature_size, 
                 optim_params, 
                 roi_res,
                 checkpoint_filename,
                 save_wrap_res,
                 save_pixel_multiply,
                 phase_path=None, 
                 prop_model='ASM',
                 propagator=None, 
                 aperture_type='square',
                 radial_symmetry=False,
                 input_power=0.0,
                 slm_resolution=None,
                 output_size=None,
                 output_resolution=None,
                 fabrication=None, 
                 min_dose=0.0, max_dose=255.0,
                 refraction_index=None,
                 normalize=True,
                 loss_params=None, 
                 writer=None, 
                 device=torch.device('cuda'), 
                 dtype=torch.float32):
        super(BS, self).__init__()
        # Setting parameters
        # convert to tensors with shape of [1,C,1,1]
        self.fab = fabrication
        self.min_dose = min_dose
        self.max_dose = max_dose
        self.refraction_index = refraction_index
        self.wavelength = wavelength
        self.prop_dist = prop_dist
        
        self.feature_size = feature_size
        self.roi_res = roi_res
        self.phase_path = phase_path
        self.precomputed_H = None
        self.prop_model = prop_model
        self.prop = propagator
        self.aperture_type = aperture_type
        self.radial_symmetry=radial_symmetry
        self.input_power = input_power
        self.slm_resolution=slm_resolution
        self.input_phase = None
        self.zfft2 = None
        self.output_size = output_size
        self.output_resolution = output_resolution
        self.filename = checkpoint_filename
        self.save_wrap_res = save_wrap_res
        self.save_pixel_multiply = save_pixel_multiply
        self.normalize=normalize

        self.loss_params = loss_params
        self.optim_params = optim_params

        self.writer = writer
        self.dev = device
        self.dtype = dtype

    def forward(self, target_amp, init_value=None):
        # Pre-compute propagataion kernel only once
        if self.precomputed_H is None:
            sz = [init_value.shape[0],init_value.shape[1],self.slm_resolution[0],self.slm_resolution[1]]
            if self.prop_model == 'ASM':
                self.precomputed_H = self.prop(torch.empty(*sz, dtype=torch.complex64), self.feature_size,
                                            self.wavelength, self.prop_dist, 
                                            self.output_resolution,
                                            return_H=True)
                self.precomputed_H = self.precomputed_H.to(self.dev).detach()
                self.precomputed_H.requires_grad = False
                self.zfft2 = None
            if self.prop_model == 'SFR':
                self.zfft2 = self.prop(torch.empty(*sz, dtype=torch.complex64, device=self.dev), self.feature_size,
                                                self.wavelength, self.prop_dist, 
                                                self.output_size,self.output_resolution,
                                                return_zfft2=True, dtype=self.dtype)
                self.precomputed_H = self.prop(torch.empty(*sz, dtype=torch.complex64), self.feature_size,
                                                self.wavelength, self.prop_dist, 
                                                self.output_size,self.output_resolution,
                                                return_H=True, zfft2=self.zfft2, dtype=self.dtype)
                self.precomputed_H = [H.to(self.dev).detach() for H in self.precomputed_H]
                for H in self.precomputed_H:
                    H.requires_grad = False
        
        if self.input_power>0 and self.input_phase is None:
            phase_shift = utils.spherical_phase(self.slm_resolution, 
                                                self.feature_size, 
                                                self.wavelength, 
                                                1.0/self.input_power)
            self.input_phase = torch.tensor(phase_shift,dtype=self.dtype).to(self.dev).detach()
        
        # Run algorithm
        final_dose,final_height,final_amp,loss_value = binary_search(init_value, target_amp, 
                                                  self.optim_params, 
                                                  self.prop_dist, 
                                                  self.wavelength, 
                                                  self.feature_size,
                                                  self.filename, 
                                                  phase_path=self.phase_path,
                                                  fab_model=self.fab, 
                                                  min_dose=self.min_dose, max_dose=self.max_dose,
                                                  refraction_index = self.refraction_index,
                                                  prop_model=self.prop_model, 
                                                  propagator=self.prop,
                                                  input_phase=self.input_phase,
                                                  slm_resolution=self.slm_resolution,
                                                  aperture_type = self.aperture_type,
                                                  radial_symmetry = self.radial_symmetry,
                                                  zfft2=self.zfft2,
                                                  output_size=self.output_size,
                                                  output_resolution=self.output_resolution,
                                                  roi_res=self.roi_res, 
                                                  normalize=self.normalize,
                                                  loss_params=self.loss_params,
                                                  writer=self.writer,
                                                  save_wrap_res=self.save_wrap_res,
                                                  save_pixel_multiply=self.save_pixel_multiply,
                                                  dtype=self.dtype,
                                                  precomputed_H=self.precomputed_H)
        return final_dose,final_height,final_amp,loss_value

    @property
    def init_scale(self):
        return self._init_scale

    @init_scale.setter
    def init_scale(self, s):
        self._init_scale = s

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path

    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self, prop):
        self._prop = prop

