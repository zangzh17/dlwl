import math
import cmath
import torch
import numpy as np
import utils.utils as utils
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


def propagation_ASM(u_in, feature_size, wavelength, z, 
                    output_resolution=None,
                    linear_conv=True,
                    padtype='zero', 
                    return_H=False, precomped_H=None,
                    dtype=torch.float32):
    """Propagates the input field using the angular spectrum method

    Inputs
    ------
    u_in: PyTorch Complex tensor (torch.cfloat) of size (num_images, 1, height, width) -- updated with PyTorch 1.7.0
    feature_size: (height, width) of individual holographic features in m
    wavelength: wavelength in m, numpy array in shape of [1,C,1,1]
    z: propagation distance, numpy array in shape of [1,C,1,1]
    linear_conv: if True, pad the input to obtain a linear convolution
    padtype: 'zero' to pad with zeros, 'median' to pad with median of u_in's
        amplitude
    return_H[_exp]: used for precomputing H or H_exp, ends the computation early
        and returns the desired variable
    precomped_H[_exp]: the precomputed value for H or H_exp
    dtype: torch dtype for computation at different precision

    Output
    ------
    tensor of size (num_images, 1, height, width)
    """
    
    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in**2).sum(-1), 0.5))
        u_in = utils.pad_image(u_in, conv_size, padval=padval)

    
    if precomped_H is None:
        # compute H
        # shape of H: [1,C,H,W]
        field_resolution = u_in.size()
        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]
        # sampling inteval size
        dy, dx = feature_size
        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))
        # critical distance
        # for z>zc, use RSC; else, use ASM
        zc_x = 2*num_x*dx**2/wavelength * np.sqrt(1-(wavelength/(2*dx))**2)
        zc_y = 2*num_y*dy**2/wavelength * np.sqrt(1-(wavelength/(2*dy))**2)
        zc = (zc_x+zc_y)/2
        print('Critical range zc: ', zc.squeeze())
        print('Propagation range z: ', z.squeeze())
        # spatial coordinates sampling
        sy = np.linspace(-y/2, y/2-dy, num_y)
        sx = np.linspace(-x/2, x/2-dx, num_x)
        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)
        # momentum/reciprocal space, reshape to [1,1,H,W]
        FX, FY = np.meshgrid(fx, fy)
        X,  Y  = np.meshgrid(sx, sy)
        FX = FX.reshape(1,1,*FX.shape)
        FY = FY.reshape(1,1,*FY.shape)
        X  = X.reshape(1,1,*X.shape)
        Y =  Y.reshape(1,1,*Y.shape)
        
        # transfer function of band-limited ASM - Matsushima et al. (2009)
        # shape: [1,C,H,W]
        HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))
        HH[np.isnan(HH)]=0
        fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength
        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)
        H_ASM = torch.exp(1j*H_exp*torch.tensor(z, dtype=dtype).to(u_in.device)) * H_filter.to(u_in.device)
        # H_ASM = torch.exp(1j*H_exp*torch.tensor(z, dtype=dtype).to(u_in.device)) 
        H_ASM = torch.fft.ifftshift(H_ASM)

        # transfer function of RSC
        R = np.sqrt(X**2 + Y**2 + z**2)
        h = 1/2/np.pi * z/R * (1/R - 1j* 2*np.pi/wavelength) * np.exp(1j*R * 2*np.pi/wavelength)/R
        # create tensor & upload to device (GPU)
        H_RSC = torch.tensor(np.fft.fft2(np.fft.fftshift(h)), dtype=H_ASM.dtype).to(u_in.device)
        H_RSC = H_RSC / ((H_RSC.abs()**2).sum().sqrt() / (H_ASM.abs()**2).sum().sqrt())
        # combine RSC/ASM according to z>zc and z<=zc
        z_filter = z<=zc
        asm_filter = torch.tensor(z_filter.astype(np.uint8), dtype=dtype).to(u_in.device)
        rsc_filter = torch.tensor((~z_filter).astype(np.uint8), dtype=dtype).to(u_in.device)
        H = asm_filter * H_ASM + rsc_filter * H_RSC

    else:
        H = precomped_H

    # return for use later as precomputed inputs
    if return_H:
        return H

    u_in = H *  torch.fft.fftn(torch.fft.ifftshift(u_in,dim=(-2, -1)), dim=(-2, -1), norm='ortho')
    u_in = torch.fft.fftshift(torch.fft.ifftn(u_in, dim=(-2, -1), norm='ortho'),dim=(-2, -1))

    if linear_conv:
        # return utils.crop_image(u_out, input_resolution) # using stacked version
        u_in = utils.crop_image(u_in, input_resolution, pytorch=True)  # using complex tensor
    
    if output_resolution is not None:
        u_in = utils.fft_interp(u_in, output_resolution)
    
    return u_in


def propagation_FFT(u_in,output_resolution=None,z=1):
    """Calculate far field using angular spectrum

    Inputs
    ------
    u_in: PyTorch Complex tensor (torch.cfloat) of size (num_images, 1, height, width) -- updated with PyTorch 1.7.0

    Output
    ------
    tensor of size (num_images, C, height, width)
    """

    # resolution of input field, should be: (num_images, num_channels, height, width)
    if max(z)>=0:
        u_in = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(u_in,dim=(-2, -1)), dim=(-2, -1), norm='ortho'),dim=(-2, -1))
    else:
        u_in = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(u_in,dim=(-2, -1)), dim=(-2, -1), norm='ortho'),dim=(-2, -1))
    # output interp
    if output_resolution is not None:
        u_in = utils.fft_interp(u_in, output_resolution)
    return u_in

def propagation_SFR(u_in, feature_size, wavelength, z,
                    output_size=None,output_resolution=None,
                    return_zfft2=False, zfft2=None,
                    return_H=False, precomped_H=None,
                    dtype=torch.float32):
    """Propagates the input field using the single Fresnel transform method
    Inputs
    ------
    u_in: PyTorch Complex tensor (torch.cfloat) of size (num_images, 1, height, width) -- updated with PyTorch 1.7.0
    feature_size: (height, width) of individual holographic features in m
    wavelength: wavelength in m
    z: propagation distance
    return_H[_exp]: used for precomputing H or H_exp, ends the computation early
        and returns the desired variable
    precomped_H[_exp]: the precomputed value for H or H_exp
    dtype: torch dtype for computation at different precision

    Output
    ------
    tensor of size (num_images, C, height, width)
    """
    
    # constants
    input_resolution = u_in.size()[-2:]
    dy, dx = feature_size
    input_size = [input_resolution[0]*dy,input_resolution[1]*dx]
    if output_size is None:
        output_size = input_size
    if output_resolution is None:
        output_resolution = input_resolution

    # initialize zoom fft2
    if zfft2 is None:
        sy = output_size[0] / (wavelength *z/dy)
        sx = output_size[1] / (wavelength *z/dx)
        ZoomFFT = utils.ZoomFFT2(input_resolution,
                                sy, sx,
                                output_resolution,
                                device=u_in.device,dtype=dtype)
                                
        L = [round((wavelength.min()*z.min()/p-W),3) for p,W in zip(feature_size,input_size)]
        print('Input res. = ', tuple(input_resolution))
        print('Output res. = ', output_resolution)
        # print('Pitch: ', feature_size)
        print('Max. Output Size of FSR: ', L)
        print('Used Output Size of FSR: ', output_size)
        zfft2 = ZoomFFT.cfft2
    if return_zfft2:
        return zfft2

    # compute inner/outer chirp
    if precomped_H is None:
        # inner chirp
        # number of pixels
        num_y, num_x = input_resolution[0], input_resolution[1]
        # spatial coordinates sampling
        ys = np.linspace(-input_size[0]/2,input_size[0]/2-input_size[0]/num_y, num_y)
        xs = np.linspace(-input_size[1]/2,input_size[1]/2-input_size[1]/num_x, num_x)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        xs_grid = xs_grid.reshape(1,1,*xs_grid.shape)
        ys_grid = ys_grid.reshape(1,1,*ys_grid.shape)
        # tensor shape: [1,C,H,W]
        H_exp = torch.tensor(np.pi/wavelength/z*(xs_grid**2+ys_grid**2), dtype=dtype)
        # get real/img components
        H_in = torch.exp(1j*H_exp).to(u_in.device)

        # outer chirp
        # number of pixels
        num_y, num_x = output_resolution[0], output_resolution[1]
        # spatial coordinates sampling
        ys = np.linspace(-output_size[0]/2,output_size[0]/2-output_size[0]/num_y,num_y)
        xs = np.linspace(-output_size[1]/2,output_size[1]/2-output_size[1]/num_x,num_x)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        ys_grid = ys_grid.reshape(1,1,*ys_grid.shape)
        xs_grid = xs_grid.reshape(1,1,*xs_grid.shape)
        
        # tensor shape: [1,C,H,W]
        H_exp = torch.tensor(np.pi/wavelength/z*(xs_grid**2+ys_grid**2), dtype=dtype)
        prop_phase = torch.tensor(2*np.pi*z/wavelength, dtype=dtype)
        
        # get real/img components
        # # with amp attenuation
        # amp = torch.tensor(1/wavelength/z, dtype=dtype)
        # H_out = (torch.exp(1j*H_exp) * amp/(1j) * torch.exp(1j*prop_phase)).to(u_in.device)
        # # w/o amp attenuation
        H_out = (torch.exp(1j*H_exp) /(1j) * torch.exp(1j*prop_phase)).to(u_in.device)
    else:
        H_in, H_out = precomped_H
     
    # return for use later as precomputed inputs
    if return_H:
        return H_in, H_out

    
    # process with zoom-FFT
    U = zfft2(torch.fft.ifftshift(H_in*u_in,dim=(-2, -1)))

    # return cropped result (with outer chirp modulation)
    return H_out*U


class fabrication_model(nn.Module):
    '''Use GT/LP model to calculate height distribution from dose
    '''
    def __init__(self, fit,
                 feature_size,
                 device=torch.device('cuda'),
                 dtype=torch.float32):
        '''
        fit: fit_model class
        feature_size:  (height, width) of individual pixels in m
        '''
        super(fabrication_model, self).__init__()
        # Fitting model
        self.model_gt = fit.gt_torch
        self.model_gt_inv = fit.gt_inv_torch
        self.model_lp = fit.lp
        self.dev = device
        self.dtype = dtype
        self.precomputed_H = None
        self.precomputed_H_b = None
        self.feature_size = feature_size
        self.dep = self.model_gt(torch.tensor(255.0,device=device)) * 1.0e-6
        # fab. model selection
        self.use_psf = True
        self.use_gt = True
        # distortion params
        self.gamma = 1.0
        self.amp = 1.0
        self.clamp = True

    def forward(self, field_in, backward=False, cutoff_MTF=0.05):
        '''
        Input
        -----
        field_in: input dose distribution
        gamma: distortion by power model

        Output
        -----
        height_out: height topograpy distribution
        '''
        # Pre-compute foward/backward kernel only once
        if self.precomputed_H is None:
            self.compute_H(field_in)
            self.compute_H_b(field_in,cutoff_MTF=cutoff_MTF)
        elif self.field_resolution != field_in.size():
            self.compute_H(field_in)
            self.compute_H_b(field_in,cutoff_MTF=cutoff_MTF)
        
        if backward:
            # backward model: height -> dose
            if self.use_gt:
                # apply inv-GT
                dose_out = self.gt_inv(field_in)
            else:
                # use linear inv-GT
                dose_out = (self.dep-field_in)/self.dep * 255.0
            if self.use_psf:
                # apply inv-PSF kernel
                    dose_out = torch.fft.fftn(dose_out,dim=(-2, -1),norm='ortho')
                    dose_out = self.precomputed_H_b * dose_out
                    dose_out = torch.fft.ifftn(dose_out,dim=(-2, -1),norm='ortho').real
            # Clamp dose to 0-255
            dose_out = torch.clamp(dose_out,0.0,255.0)
            return dose_out
        else:
            # forward model: dose -> height
            # Clamp dose to 0-255
            field_in = torch.clamp(field_in,0.0,255.0)
            if self.use_psf:
                # apply PSF kernel
                    field_in = torch.fft.fftn(field_in,dim=(-2, -1),norm='ortho')
                    field_in = self.precomputed_H * field_in
                    field_in = torch.fft.ifftn(field_in,dim=(-2, -1),norm='ortho').real
            if self.use_gt:
                # apply GT
                height_out = self.gt(field_in)
            else:
                # use linear GT
                height_out = self.dep - (field_in/255.0) * self.dep
            return height_out
    
    def gt(self,field_in):
        # dose->height
        # apply GT, unit conversion from um to m
        depth_out = self.model_gt(field_in) * 1.0e-6
        # apply distortion
        depth_out = (depth_out/self.dep)**self.gamma * self.dep * self.amp
        if self.clamp:
            depth_out = torch.clamp(depth_out,0.0,self.dep)
        # depth to height
        height_out = self.dep-depth_out
        return height_out
        
    def gt_inv(self,height_in):
        # height->dose
        # height to depth
        depth_in = self.dep-height_in
        # apply inv GT, unit conversion from m to um
        return self.model_gt_inv(depth_in * 1.0e6)

    def compute_H(self,field_in):
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        self.field_resolution = field_in.size()
        # number of pixels
        num_y, num_x = self.field_resolution[2], self.field_resolution[3]
        dy, dx = self.feature_size
        # frequency coordinates sampling
        fy = np.fft.fftfreq(num_y,dy)
        fx = np.fft.fftfreq(num_x,dx)
        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)
        # unit conversion from m^-1 to um^-1
        F2 = np.sqrt(FX**2 + FY**2) *1e-6
        self.precomputed_H = torch.tensor(self.model_lp(F2),dtype=self.dtype)[None,None,:,:]
        self.precomputed_H = self.precomputed_H.to(self.dev).detach()
        self.precomputed_H.requires_grad = False
    def compute_H_b(self,field_in,cutoff_MTF=0.05):
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        self.field_resolution = field_in.size()
        # number of pixels
        num_y, num_x = self.field_resolution[2], self.field_resolution[3]
        dy, dx = self.feature_size
        # frequency coordinates sampling
        fy = np.fft.fftfreq(num_y,dy)
        fx = np.fft.fftfreq(num_x,dx)
        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)
        # unit conversion from m^-1 to um^-1
        F2 = np.sqrt(FX**2 + FY**2) *1e-6
        # MTF value at different freq
        M = self.model_lp(F2)
        # H_filter = torch.tensor((M<cutoff_MTF).astype(np.uint8), dtype=self.dtype)[None,None,:,:]
        M[M<cutoff_MTF] = 1.0
        M[num_y//2, num_x//2] = 1.0
        self.precomputed_H_b = torch.tensor(1/M,dtype=self.dtype)[None,None,:,:]
        self.precomputed_H_b = self.precomputed_H_b.to(self.dev).detach()
        self.precomputed_H_b.requires_grad = False