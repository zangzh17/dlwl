"""
This is the script containing all uility functions used for the implementation.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""
import math
import cmath
import random
from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as ll

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2



class ZoomFFT2:
    """Compute the chirp z-transform to achieve scaled FFT
    input: [B,C,H,W]
    
    dim 'C' is used for multi-channel processing

    Parameters
    ----------
    m : int, optional
        The number of points to evaluate.  
        The default is the length of `x`.
    sy, sx: numpy array 
            frequency zoom factor for each channel
            array([sy1,sy2,...])
            array([sx1,sx2,...])

    """
    def __init__(self, n, sy, sx, m=None,  
                 device=torch.device('cuda'),dtype=torch.float32):
        if m is None:
            m = n
        num_channel = sy.size
        if sy.size != sx.size:
            ValueError('channel number is not equal for x and y')
        sx = sx.ravel()[None,:,None,None]
        sy = sy.ravel()[None,:,None,None]

        num_y, num_x = n[0], n[1]
        num_v, num_u = m[0], m[1]
        # used for normalization
        self.N = torch.tensor(num_v* num_u).to(device=device,dtype=dtype)

        wx = -2*np.pi * sx / num_u
        wy = -2*np.pi * sy / num_v
        ax =  2*np.pi * (-sx/2)
        ay =  2*np.pi * (-sy/2)

        kky = (np.arange(-num_y+1,max(num_y,num_v))[None,None,:,   None]**2)/2
        kkx = (np.arange(-num_x+1,max(num_x,num_u))[None,None,None,:   ]**2)/2
        nny = np.arange(0,num_y)[None,None,:,   None]
        nnx = np.arange(0,num_x)[None,None,None,:   ]

        ww = wx * kkx + wy * kky
        aa = ax * (-nnx) + ay * (-nny) + ww[:,:,(num_y-1):(2*num_y-1),(num_x-1):(2*num_x-1)]
        
        nfft = (int(2**np.ceil(np.log2(num_v+num_y-1))),
                int(2**np.ceil(np.log2(num_u+num_x-1))))

        yidx = slice(num_y-1, num_y+num_v-1)
        xidx = slice(num_x-1, num_x+num_u-1)

        # upload
        # initial chirp: A
        # FFT kernel: W
        # IFFT kernel: iW
        # outer chirp: B
        # correction: C
        A = torch.from_numpy(aa).to(device=device,dtype=dtype)
        self._A = torch.exp(1j*A)
        B = torch.from_numpy(ww[:,:,(num_y-1):(num_v+num_y-1), (num_x-1):(num_u+num_x-1)]).to(device=device,dtype=dtype)
        self._B = torch.exp(1j*B)

        W = torch.from_numpy(-ww[:,:,:(num_v+num_y-1),:(num_u+num_x-1)]).to(device=device,dtype=dtype) 
        self._W = torch.fft.fft2(torch.exp(1j*W), nfft, norm='ortho') 
        self._iW = torch.fft.fft2(torch.exp(-1j*W), nfft, norm='ortho')

        self._yidx = yidx
        self._xidx = xidx
        self._nfft = nfft

        # used for phase correction in centered FFT/IFFT
        C = np.zeros((1,num_channel,num_v,num_u))
        for i,(ry,rx) in enumerate(zip(sy.flatten(),sx.flatten())):
            fx = np.linspace(-rx/2, rx/2-rx/num_u, num_u)*(int((num_x-1)/2)+0.5)
            fy = np.linspace(-ry/2, ry/2-ry/num_v, num_v)*(int((num_y-1)/2)+0.5)
            C[0,i,:,:] = fx[None,:] + fy[:,None]
        self._C = torch.from_numpy(C).to(device=device,dtype=dtype)
        self._C = torch.exp(1j * self._C * 2*torch.pi)

    def fft2(self, x):
        """
        Calculate the zoomed FFT:

        \sum_{n=0}^{N-1} x_n exp(-j2pi*f*n) 
        """
        
        y = torch.fft.ifft2(self._W * torch.fft.fft2(x * self._A, self._nfft,norm='ortho'),norm='ortho')
        y = y[..., self._yidx, self._xidx] * self._B 
        return y

    def ifft2(self, x):
        """
        Calculate the zoomed iFFT:
        \sum_{k=0}^{K-1} X_k exp(j2pi*n*fk) 
        
        """
        y = torch.fft.ifft2(self._iW * torch.fft.fft2(x / self._A, self._nfft,norm='ortho'),norm='ortho')
        y = y[..., self._yidx, self._xidx] / self._B
        return y
    def cifft2(self,x):
        y = self.ifft2(torch.fft.fftshift(x,dim=(-2,-1))) / self._C
        return y
    def cfft2(self,x):
        y = self.fft2(torch.fft.fftshift(x,dim=(-2,-1))) * self._C
        return y
    



        

def pad_image(field, target_shape, pytorch=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    size_diff = np.array(target_shape) - np.array(field.shape[-2:])
    odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2
        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if mode == 'constant':
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
            elif mode == 'wrap' or mode == 'circular':
                return nn.functional.pad(field, pad_axes, mode='circular')
            else:
                return nn.functional.pad(field, pad_axes, mode=mode)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            if mode == 'constant':
                return np.pad(field, tuple(zip(pad_front, pad_end)), mode, 
                          constant_values=padval)
            elif mode == 'wrap' or mode == 'circular':
                return np.pad(field, tuple(zip(pad_front, pad_end)), mode='wrap') 
            else:
                return np.pad(field, tuple(zip(pad_front, pad_end)), mode)
    else:
        return field


def crop_image(field, target_shape, pytorch=True, normalize=False):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field
    
    size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
    odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if normalize:
            field_cropped = field[(..., *crop_slices)]
            return field_cropped / ((abs(field_cropped)**2).sum()**0.5) * ((abs(field)**2).sum()**0.5) 
        else:
            return field[(..., *crop_slices)]
    else:
        return field

def crop_pad_image(field, target_shape, pytorch=True,normalize=False):
    """Crops and/or pad a 2D field, according to target_shape
    """
    if target_shape is None:
        return field
    field = crop_image(field, target_shape, pytorch=pytorch, normalize=normalize)
    field = pad_image(field, target_shape, pytorch=pytorch)
    return field

def crop_mask(field, roi_res, pytorch=True):
    """cropping from image_res of field to roi_res
    output: index (mask) on the initial image
    """
    field_shape = np.array(field.shape)
    if roi_res is None:
        crop_slices = [slice(w) for w in field_shape[-2:]]
    else:

        size_diff = np.array(field_shape[-2:]) - np.array(roi_res)
        odd_dim = np.array(field_shape[-2:]) % 2

        # crop dimensions that need to decrease in size
        if (size_diff > 0).any():
            crop_total = np.maximum(size_diff, 0)
            crop_front = (crop_total + 1 - odd_dim) // 2
            crop_end = (crop_total + odd_dim) // 2

            crop_slices = [slice(int(f), int(-e) if e else None)
                        for f, e in zip(crop_front, crop_end)]
        else:
            crop_slices = [slice(w) for w in field_shape[-2:]]
    front_slices = [slice(w) for w in field_shape[:-2]]      
    return (*front_slices,*crop_slices)

def loss(recon_amp,recon_phase,target_amp,
         loss_params,
         cutoff_frequency=0.5,
         roi_res=None,
         s=1.0):
    # generate loss value between 'target_amp' and 'loss_params' 
    #   according to 'loss_params'
    # recon_amp will be cropped by roi_res
    # assuming target_amp has already been cropped to the size of 'roi_res' (for speed up)
    # s: learnable scaling parameter

    # loss fcn
    if loss_params.custom == 'L2':
        custom_loss = nn.MSELoss()
    elif loss_params.custom == 'L1':
        custom_loss = nn.L1Loss()
    
    
    # crop recon_amp
    if roi_res is not None:
        recon_amp = crop_image(recon_amp,roi_res)
    
    # add mask
    # zero_ord_idx = [cfftindex(N) for N in target_amp.shape[-2:]]
    nzero_idx = target_amp!=0
    if loss_params.use_nonzero_mask:
        mask = nzero_idx
    elif loss_params.use_zero_mask:
        mask = ~nzero_idx
    else:
        mask = recon_amp==recon_amp

    # calc. weighted intensity loss    
    weight = loss_params.channel_weight*loss_params.batch_weight
    if loss_params.square:
        recon_amp = recon_amp**2
        target_amp = target_amp**2

    lossValue = 0
    # # phase constraint (spectrum panalty)
    # if loss_params.laplace_penalty_weight>0:
    #     lossValue = lossValue + loss_params.laplace_penalty_weight*laplacian(recon_phase).abs().sum()
    if loss_params.spectrum_penalty_weight>0:
        lossValue = lossValue + loss_params.spectrum_penalty_weight*spectrum_penalty(torch.exp(1j*recon_phase),loss_params.spectrum_penalty_radial_weight,cutoff_frequency)

    # add efficiency penalty
    # if loss_params.eff_penalty_weight>0:
    #     effPenalty = -0.1 * (recon_amp * target_amp *weight).sum()/(recon_amp).sum()
    #     lossValue = lossValue + effPenalty*loss_params.eff_penalty_weight
    if loss_params.eff_penalty_weight>0:
        effPenalty = efficiency_penalty(recon_amp, cutoff_ratio=loss_params.eff_penalty_ratio)
        lossValue = lossValue + effPenalty*loss_params.eff_penalty_weight

    # normalize
    recon_amp = normalize(recon_amp,method=loss_params.normalize)
    target_amp = normalize(target_amp,method=loss_params.normalize)

    
    if loss_params.custom == 'None':
        pass
    elif loss_params.custom == 'L2':
        lossValue = lossValue + custom_loss(s * recon_amp[mask]*weight, 
                             target_amp[mask]*weight).sqrt()
    elif loss_params.custom == 'L1':
        lossValue = lossValue + custom_loss(s * recon_amp[mask]*weight, 
                             target_amp[mask]*weight)

    # add variation penalty
    if loss_params.var_penalty_weight>0:
        mean = recon_amp[nzero_idx].mean().expand(recon_amp[nzero_idx].shape)
        varPenalty = nn.MSELoss(s * recon_amp[nzero_idx]*weight, 
                                  mean*weight).sqrt()
        lossValue = lossValue + loss_params.var_penalty_weight * varPenalty

    
    return lossValue



def efficiency_penalty(input_power, cutoff_ratio=0.5):
    _, _, h, w = input_power.shape
    
    # Create a mask for peri-components
    fy = int((cutoff_ratio*h)//2)
    fx = int((cutoff_ratio*w)//2)
    mask = torch.ones_like(input_power)
    mask[..., h//2-fy:h//2+fy, w//2-fx:w//2+fx] = 0
    
    # # Create a mask for centered components
    # fy = int((cutoff_ratio*h)//2)
    # fx = int((cutoff_ratio*w)//2)
    # mask = torch.zeros_like(input_power)
    # mask[..., h//2-fy:h//2+fy, w//2-fx:w//2+fx] = 1

    # # radial weight
    # y, x = np.mgrid[:h, :w]
    # cy, cx = h // 2, w // 2
    # dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    # dist[..., h//2-fy:h//2+fy, w//2-fx:w//2+fx] = 0
    # max_dist = np.max(dist)
    # normalized_dist = dist / max_dist
    # radial_weight = (1 - normalized_dist)**2 
    # radial_weight = torch.tensor(radial_weight, dtype=input_power.dtype, device=input_power.device)
    # return -torch.sum(input_power * mask * radial_weight)/torch.sum(input_power*radial_weight)
    return torch.sum(input_power.abs() * mask)

def spectrum_penalty(input, weight=10, cutoff_frequency=0.5):
    # Compute the FFT of the input and target tensors
    input_fft = torch.fft.fftshift(torch.fft.fft2(input)).abs()**2
    _, _, h, w = input_fft.shape

    # # apply radial weight
    # y, x = np.mgrid[:h, :w]
    # cy, cx = h // 2, w // 2
    # dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    # max_dist = np.max(dist)
    # normalized_dist = dist / max_dist
    # radial_weight = weight * normalized_dist
    # radial_weight = torch.tensor(radial_weight, dtype=input_fft.dtype, device=input_fft.device)
    
    # Create a mask for high-frequency components
    fy = int((cutoff_frequency*h)//2)
    fx = int((cutoff_frequency*w)//2)
    mask = torch.ones_like(input_fft)
    mask[..., h//2-fy:h//2+fy, w//2-fx:w//2+fx] = 0
    
    # # Create a mask for low-frequency components
    # fy = int((cutoff_frequency*h)//2)
    # fx = int((cutoff_frequency*w)//2)
    # mask = torch.zeros_like(input_fft)
    # mask[..., h//2-fy:h//2+fy, w//2-fx:w//2+fx] = 1

    # return torch.sum(input_fft * mask * radial_weight)/torch.sum(input_fft*radial_weight)
    return torch.sum(input_fft * mask)/torch.sum(input_fft)

def laplacian(img):
    # signed angular difference
    grad_x1, grad_y1 = grad(img, next_pixel=True)  # x_{n+1} - x_{n}
    grad_x0, grad_y0 = grad(img, next_pixel=False)  # x_{n} - x_{n-1}

    laplacian_x = grad_x1 - grad_x0  # (x_{n+1} - x_{n}) - (x_{n} - x_{n-1})
    laplacian_y = grad_y1 - grad_y0

    return laplacian_x + laplacian_y


def grad(img, next_pixel=False, sovel=False):
    
    if img.shape[1] > 1:
        permuted = True
        img = img.permute(1, 0, 2, 3)
    else:
        permuted = False
    
    # set diff kernel
    if sovel:  # use sovel filter for gradient calculation
        k_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32) / 8
        k_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32) / 8
    else:
        if next_pixel:  # x_{n+1} - x_n
            k_x = torch.tensor([[0, -1, 1]], dtype=torch.float32)
            k_y = torch.tensor([[1], [-1], [0]], dtype=torch.float32)
        else:  # x_{n} - x_{n-1}
            k_x = torch.tensor([[-1, 1, 0]], dtype=torch.float32)
            k_y = torch.tensor([[0], [1], [-1]], dtype=torch.float32)

    # upload to gpu
    k_x = k_x.to(img.device).unsqueeze(0).unsqueeze(0)
    k_y = k_y.to(img.device).unsqueeze(0).unsqueeze(0)

    # boundary handling (replicate elements at boundary)
    img_x = F.pad(img, (1, 1, 0, 0), 'replicate')
    img_y = F.pad(img, (0, 0, 1, 1), 'replicate')

    # take sign angular difference
    grad_x = signed_ang(F.conv2d(img_x, k_x))
    grad_y = signed_ang(F.conv2d(img_y, k_y))
    
    if permuted:
        grad_x = grad_x.permute(1, 0, 2, 3)
        grad_y = grad_y.permute(1, 0, 2, 3)

    return grad_x, grad_y


def radial_symmetric_interpolation(radial_profile, shape=None, mode='nearest'):
    # Performs radial symmetric interpolation of a given radial profile.
    # The function uses grid_sample function of PyTorch to interpolate the radial profile onto a polar grid.
    # The resulting 2D tensor contains the interpolated values in a symmetric pattern.
    # The function takes three arguments:
    #   - radial_profile: the input radial 1D intensity profile, shape [B,C,H,1]
    #   - shape: 4-element tuple denoting [B,C,H,W]. shape of the output tensor (optional)
    #   - mode: the interpolation mode (optional, default is 'nearest')
    # If shape is not provided, the output shape is determined by the length of the radial_profile.
    # The resulting 4D tensor is returned as the output of the function.
    device = radial_profile.device
    dtype = radial_profile.dtype
    if shape is None:
        shape = (1,1,radial_profile.shape[2]*2-1,radial_profile.shape[2]*2-1)
    B, C, H, W = shape

    # create_polar_meshgrid(H, W):
    xc, yc = W // 2, H // 2
    y = torch.arange(H).to(dtype=dtype)
    x = torch.arange(W).to(dtype=dtype)
    x_grid, y_grid = torch.meshgrid(x - xc, y - yc,indexing='xy')
    r_grid = torch.sqrt(x_grid**2 + y_grid**2)
    r_grid = r_grid.view(1, H, W).to(dtype=dtype)

    # Normalize the polar grid to match the range of radial_profile 
    # as grid for grid_sample should be in the range [-1, 1]
    r_grid_normalized = 2 * r_grid / r_grid.max()  - 1

    grid = torch.stack([-1*torch.ones_like(r_grid_normalized),
                        r_grid_normalized
                        ], dim=-1).repeat(B, 1, 1, 1).to(device=device)

    # Interpolate by using grid_sample
    symmetric_2d = F.grid_sample(radial_profile, grid,mode=mode, align_corners=True)
    return symmetric_2d

def signed_ang(angle):
    """
    cast all angles into [-pi, pi]
    """
    return (angle + math.pi) % (2*math.pi) - math.pi


def circular_clamp(x, a, b):
    # calculate the range length
    range_len = b - a
    # calculate the circularly clamped value
    clamped_val = torch.where(x < a, 
                              b - (a-x-1)%range_len, 
                              torch.where(x > b, a + (x-b-1)%range_len, x))
    return clamped_val

def tile_batch(input, layout, pytorch=True,uint8=False):
    batch_num = input.shape[0]
    layout = list(layout)
    if layout[0]==-1:
        layout[0] = int(np.ceil(batch_num/layout[1]))
    elif layout[1]==-1:
        layout[1] = int(np.ceil(batch_num/layout[0]))
    
    input_res = input.shape[-2:]
    output_res = [N*r for N,r in zip(input_res,layout)]
    if pytorch:
        if uint8:
            output = torch.zeros(1,1,*output_res,dtype=torch.uint8)
        else:
            output = torch.zeros(1,1,*output_res)
    else:
        if uint8:
            output = np.zeros((1,1,*output_res),dtype=np.uint8)
        else:
            output = np.zeros((1,1,*output_res))
    for i in range(layout[0]):
        for j in range(layout[1]):
            k = i*layout[0]+j
            output[0,0, i*input_res[0]:(i+1)*input_res[0],
                        j*input_res[1]:(j+1)*input_res[1]] = input[k,0,:,:]
    return output
    
def calculate_ratio(input_field, cutoff_ratio=0.1,pytorch=True):
    if pytorch:
        h = input_field.shape[-2]
        w = input_field.shape[-1]
        y = int((cutoff_ratio*h)//2)
        x = int((cutoff_ratio*w)//2)
        mask = torch.zeros_like(input_field)
        mask[..., h//2-y:h//2+y, w//2-x:w//2+x] = 1
        return torch.sum(mask*input_field.abs())/torch.sum(input_field.abs())
    else:
        h = input_field.shape[-2]
        w = input_field.shape[-1]
        y = int((cutoff_ratio * h) // 2)
        x = int((cutoff_ratio * w) // 2)
        mask = np.zeros_like(input_field)
        mask[..., h // 2 - y:h // 2 + y, w // 2 - x:w // 2 + x] = 1
        return np.sum(mask * np.abs(input_field),axis=(-2,-1)) / np.sum(np.abs(input_field),axis=(-2,-1))

def fft_interp(u_in, interp_resolution,pytorch=True,normalize=True):
    if (np.array(u_in.shape[-2:])==np.array(interp_resolution[-2:])).all():
        return u_in
    # use double FT to implement inerpolation
    if pytorch:
        # FT
        U = torch.fft.fftshift(torch.fft.fftn(u_in,dim=(-2, -1),norm='ortho'),dim=(-2, -1))
        # pad or crop
        U = pad_image(U,interp_resolution)
        U = crop_image(U,interp_resolution)
        # IFT
        U = torch.fft.ifftn(torch.fft.ifftshift(U,dim=(-2, -1)),dim=(-2, -1),norm='ortho')
        if not u_in.is_complex():
            U = U.real
        # normalize
        if normalize:
            U = U * torch.sqrt((u_in.abs()**2).sum()) / torch.sqrt((U.abs()**2).sum())
    else:
        # FT
        U = np.fft.fftshift(np.fft.fftn(u_in,axes=(-2, -1),norm='ortho'),axes=(-2, -1))
        # pad or crop
        U = pad_image(U,interp_resolution,pytorch=False)
        U = crop_image(U,interp_resolution,pytorch=False)
        # IFT
        U = np.fft.ifftn(np.fft.ifftshift(U,axes=(-2, -1)),axes=(-2, -1),norm='ortho')
        if not np.iscomplex(u_in).any():
            U = U.real
        # normalize
        if normalize:
            U = U * np.sqrt((np.abs(u_in)**2).sum()) / np.sqrt((np.abs(U)**2).sum())
    return U

def srgb_gamma2lin(im_in, pytorch=False):
    """converts from sRGB to linear color space"""
    thresh = 0.04045
    if pytorch:
        im_out = torch.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    else:
        im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    return im_out


def srgb_lin2gamma(im_in, pytorch=False):
    """converts from linear to sRGB color space"""
    thresh = 0.0031308
    if pytorch:
        im_out = torch.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    else:
        im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def phasemap_8bit(phasemap, inverted=True):
    """convert a phasemap tensor into a numpy 8bit phasemap that can be directly displayed

    Input
    -----
    :param phasemap: input phasemap tensor, which is supposed to be in the range of [-pi, pi].
    :param inverted: a boolean value that indicates whether the phasemap is inverted.

    Output
    ------
    :return: output phasemap, with uint8 dtype (in [0, 255])
    """

    output_phase = ((phasemap + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    else:
        phase_out_8bit = ((output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    return phase_out_8bit

def fftorder(N):
    return np.round(np.fft.fftfreq(N)*N)
def fftfreq(N):
    return np.fft.fftfreq(N)
def fftindex(N, order_number=0):
    # find the index of order_number in list of fftorders
    return np.nonzero(fftorder(N)==order_number)[0].item()
def cfftorder(N):
    return np.round(np.fft.fftshift(np.fft.fftfreq(N))*N)
def cfftfreq(N):
    return np.fft.fftshift(np.fft.fftfreq(N))
def cfftindex(N, order_number=0):
    # find the index of order_number in list of cfftorders
    return np.nonzero(cfftorder(N)==order_number)[0].item()

# def k_cartesian(Nd):
#     dim = len(Nd) # dimension
#     M = np.prod(Nd)
#     om = np.zeros((dim,M), dtype = np.float)
#     grid = np.indices(Nd)
#     for dimid in range(0, dim):
#         om[dimid,:] = (grid[dimid].ravel() *2/ Nd[dimid] - 1.0)*np.pi
#     return om
        

def burst_img_processor(img_burst_list):
    img_tensor = np.stack(img_burst_list, axis=0)
    img_avg = np.mean(img_tensor, axis=0)
    return im2float(img_avg)  # changed from int8 to float32


def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')

def amp2im(amp):
    """convert amplitude image (assuming range scaled to 0-1) to uint8
    :param im: image
    :param dtype: default np.float32
    :return:
    """
    # quantized to 8 bits
    return (amp**2*255).round().cpu().detach().squeeze().numpy().astype(np.uint8)

def pow2db(pow, db_min=-20):
    pow = pow/pow.max()
    return np.where(pow<=10**(db_min/10),db_min, 10*np.log10(pow))

def mask_gradients(grad, percentage=0.5):
    mask = torch.zeros(grad.shape, device=grad.device)
    perm = torch.randperm(mask.numel())
    mask.view(-1)[perm[:int(mask.numel()*percentage)]] = 1.0
    return mask

def round_masked(input_field):
    _,_, h, w = input_field.shape
    y, x = torch.meshgrid(torch.arange(h,device=input_field.device), 
                            torch.arange(w,device=input_field.device),
                            indexing='ij')
    x = x.float() - h // 2
    y = y.float() - w // 2
    dist = torch.sqrt(x**2 + y**2).expand(input_field.shape[0],input_field.shape[1],-1,-1)
    return torch.where(dist > min(h, w) // 4,
                        torch.tensor([0.0],device=input_field.device), 
                        input_field)

def forward_model(input_field, 
                  opt_params, 
                  propagator=None, 
                  zfft2=None,
                  fab_model=None, 
                  input_phase=None,
                  normalize=False,
                  dtype=torch.float32):
    # dose->height (w/ fab_model)
    # height->height (for none fab_model)
    input_field = fabrication_field(input_field, 
                                    aperture_type='square',
                                    fab_model=fab_model, 
                                    fab_backward=False)
    # height->comp (w/ propagator)
    # height->height (for none propagator)
    if propagator is None:
        input_field = propagate_field(input_field, 
                    propagator, 
                    prop_dist=opt_params.prop_dist, 
                    wavelength=opt_params.wavelength, 
                    refraction_index=opt_params.refraction_index,
                    feature_size=opt_params.feature_size,
                    prop_model='None',
                    prop_backward=False, 
                    input_phase=input_phase, 
                    aperture_type='square',
                    zfft2=zfft2,
                    output_size=opt_params.output_size,
                    output_resolution=opt_params.output_res,
                    normalize=normalize,
                    field_type='height',
                    dtype=dtype, 
                    precomputed_H=None)
    else:
        input_field = propagate_field(input_field, 
                    propagator, 
                    prop_dist=opt_params.prop_dist, 
                    wavelength=opt_params.wavelength, 
                    refraction_index=opt_params.refraction_index,
                    feature_size=opt_params.feature_size,
                    prop_model=opt_params.prop_model,
                    prop_backward=False, 
                    input_phase=input_phase, 
                    aperture_type='square',
                    zfft2=zfft2,
                    output_size=opt_params.output_size,
                    output_resolution=opt_params.output_res,
                    normalize=normalize,
                    field_type='height',
                    dtype=dtype, 
                    precomputed_H=None)
    
    return input_field

def height2phase(height,
                 wavelength=np.array([520e-9]).reshape(1,1,1,1), 
                 refraction_index=np.array([1.52]).reshape(1,1,1,1),
                 dtype=torch.float32):
    # height to phase
    phase = height*\
        (torch.from_numpy(refraction_index).to(dtype=dtype,device=height.device)-1)\
            /torch.from_numpy(wavelength).to(dtype=dtype,device=height.device) \
            * 2 * torch.pi
    return phase

def phase2height(phase,
                 wavelength=np.array([520e-9]).reshape(1,1,1,1), 
                 refraction_index=np.array([1.52]).reshape(1,1,1,1),
                 dtype=torch.float32):
    # phase: [B,C,H,W]  
    # -->> height: [B,1,H,W]
    # phase to height，use average strategy
    height = phase / (2*torch.pi) \
                * torch.from_numpy(wavelength).to(dtype=dtype,device=phase.device) \
                / (torch.from_numpy(refraction_index).to(dtype=dtype,device=phase.device)-1)
    # use average strategy to reduce input shape [B,C,H,W] to [B,1,H,W]
    return torch.mean(height,1,keepdim=True)

def propagate_field(input_field, 
                    propagator, 
                    prop_dist=np.array([0.2]).reshape(1,1,1,1), 
                    wavelength=np.array([520e-9]).reshape(1,1,1,1), 
                    refraction_index=np.array([1.52]).reshape(1,1,1,1), 
                    feature_size=(1.0e-6, 1.0e-6),
                    prop_model='ASM', 
                    prop_backward=False,
                    input_phase=None, 
                    aperture_type='square',
                    zfft2=None,
                    output_size=None,
                    output_resolution=None,
                    normalize=False,
                    field_type='height',
                    dtype=torch.float32, 
                    precomputed_H=None):
    """
    A wrapper for various propagation methods, including the parameterized model.
    Note that input_field is supposed to be in Cartesian coordinate, not polar!
    Forward prop.:
        input: complex field / phase / height / any (for prop_model == 'None')
        output: propagated complex field  / no change (for prop_model == 'None')
    Backward prop:
        input: complex field / phase / any (for prop_model == 'None')
        output: height  / no change (for prop_model == 'None')

    Input
    -----
    :param input_field: pytorch complex tensor shape of (1, 1, H, W), the field before model, in X, Y coordinates
    :param prop_dist: propagation distance in m. in np.array, shape:(1,C,1,1)
    :param wavelength: wavelength of the wave in m. in np.array, shape:(1,C,1,1)
    :param feature_size: pixel pitch in length of 2, tuple or list
    :param prop_model: propagation model ('ASM', 'MODEL', 'fresnel', ...)
    :param field_type: 'height' for height field, 'comp' for complex amplitude, 'phase' for optical phase
    :param dtype: torch.float32 by default
    :param precomputed_H: Propagation Kernel in Fourier domain (could be calculated at the very first time and reuse)

    Output
    -----
    :return: output_field: pytorch complex tensor shape of (1, C, H, W), the field after propagation, in X, Y coordinates
    """
    device = input_field.device

    # apply aperture mask
    if aperture_type == 'round':
        input_field = round_masked(input_field)

    # no change w/o prop_model
    if prop_model == 'None':
        return input_field

    if prop_backward:
    # Backward prop:
    #     input: complex field / phase / any for no change (for prop_model == 'None')
    #     output: height  / no change (for prop_model == 'None')
        if field_type == 'phase':
            # phase to complex amp
            input_field = torch.exp(input_field * 1j)

        if prop_model == 'ASM':
            output_field = propagator(u_in=input_field, 
                                    feature_size=feature_size,
                                    wavelength=wavelength, z=prop_dist,
                                    output_resolution = output_resolution, 
                                    dtype=dtype, 
                                    precomped_H=precomputed_H)
        elif prop_model == 'FFT':
            output_field = propagator(u_in=input_field,
                                    output_resolution=output_resolution,
                                    z=prop_dist)
        elif prop_model == 'SFR':
            output_field = propagator(u_in=input_field, 
                                        feature_size=feature_size,
                                        wavelength=wavelength, z=prop_dist,
                                        output_size=output_size,
                                        output_resolution=output_resolution,
                                        precomped_H=precomputed_H,
                                        zfft2=zfft2,
                                        dtype=dtype)
        else:
            raise ValueError('Unexpected prop_model value...')

        # phase extraction
        if input_phase is not None:
            # with init. phase: non-plane-wave illumination
            # get phase
            output_field = (torch.fmod(output_field.angle()-input_phase +torch.pi, 2 * torch.pi)+torch.pi)
        else:
            # w/o initial phase: plane wave illumination
            # get phase 
            output_field = (output_field.angle()+torch.pi)

        # phase to height，use average strategy
        # [B,C,H,W] -->> [B,1,H,W]
        output_field = phase2height(output_field,wavelength,refraction_index,dtype=dtype)

        return output_field
    else:
        # Forward prop.:
        #     input: complex field / phase / height / any for no change (for prop_model == 'None')
        #     output: propagated complex field  / no change (for prop_model == 'None')
        
        if field_type == 'phase':
            if input_phase is not None:
                # add precomputed input phase
                # can be used to simluate spherical illumination
                input_field = input_field + input_phase
            # phase to complex amp
            input_field = torch.exp(input_field * 1j)
        elif field_type == 'height':
            # height to phase
            input_field = height2phase(input_field,wavelength,refraction_index,dtype=dtype)
            if input_phase is not None:
                # add precomputed input phase
                # can be used to simluate spherical illumination
                input_field = input_field + input_phase
            # phase to complex amp
            input_field = torch.exp(input_field * 1j)
        elif field_type == 'comp': 
            if input_phase is not None:
                # add precomputed input phase
                # can be used to simluate spherical illumination
                input_field = input_field * torch.exp(input_phase * 1j)
        else:
            raise ValueError('Unexpected field_type value...')

        # diffraction propagation model
        if prop_model == 'ASM':
            output_field = propagator(u_in=input_field, 
                                    feature_size=feature_size,
                                    wavelength=wavelength, z=prop_dist,
                                    output_resolution = output_resolution, 
                                    dtype=dtype, 
                                    precomped_H=precomputed_H)
        elif prop_model == 'FFT':
            output_field = propagator(u_in=input_field,
                                    output_resolution=output_resolution,
                                    z=prop_dist)
        elif prop_model == 'SFR':
            output_field = propagator(u_in=input_field, 
                                        feature_size=feature_size,
                                        wavelength=wavelength, z=prop_dist,
                                        output_size=output_size,
                                        output_resolution=output_resolution,
                                        precomped_H=precomputed_H,
                                        zfft2=zfft2,
                                        dtype=dtype)
        else:
            raise ValueError('Unexpected prop_model value...')

        if normalize: 
            output_field = output_field/\
                            torch.sum((output_field.abs())**2,dim=(-2,-1),keepdim=True).sqrt()
                
        return output_field


def fabrication_field(input_field, 
                    aperture_type='square',
                    fab_model=None, 
                    fab_backward=False,
                    cutoff_MTF=0.2):
    """
    A wrapper for various fabrication methods, including the parameterized model.
    Note that input_field is supposed to be in Cartesian coordinate, not polar!

    Inputs:
        wavelength, numpy array with shape of [1,C,1,1]
        refraction_index, numpy array with shape of [1,C,1,1]

    Forward fab model (fab_backward=False):
    input shape [B,1,H,W]; output shape [B,C,H,W]
        W/o fab model (fab_model is None):
            [input: height] ->> [output: height] (do not change)
        W/ fab model:
            [input: dose]   ->> [output: height]
    Backward fab model (fab_backward=True):
    input shape [B,C,H,W]; output shape [B,1,H,W]
        input: height (shape [B,C,H,W]) ->>
        W/o fab model (fab_model is None): 
            [input: height] ->> [output: height] (do not change)
        W/ fab model: 
            [input: height] ->> [output: dose]
    """

    # apply aperture mask
    if aperture_type == 'round':
        input_field = round_masked(input_field)

    # W/o fab model 
    # [input: height] ->> [output: height] (do not change)
    if fab_model is None:
        return input_field

    if fab_backward:
        # Backward fab model (fab_backward=True):
        # input shape [B,1,H,W]
        # W/ fab model: 
        #     [input: height] ->> [output: dose]
        output_field = fab_model(input_field,backward=True,cutoff_MTF=cutoff_MTF)
    else:
        # Forward fabrication model
        # W/ fab model:
        #     [input: dose]   ->> [output: height]
        output_field = fab_model(input_field)

    return output_field

def spherical_phase(array_shape, sample_pitch, wavelength, focal_length):
    """
    Generates a numpy array for the phase of a spherical wave
    at the position of focal_length

    Args:
        sample_pitch (tuple): Sampling pitch in meters.
        focal_length (float): focal_length in meters.
        wavelength (numpy array: [1,C,1,1]): wavelength in each channels.
        tensor_shape (tuple): Shape of the output tensor 
                            (height, width) or (B,C,height, width)
    """
    # Create a meshgrid of x and y coordinates
    x_range = np.arange(array_shape[-1])
    y_range = np.arange(array_shape[-2])
    x_coords, y_coords = np.meshgrid(x_range, y_range)

    # Convert coordinates to meters
    x_coords = (x_coords - array_shape[-1] // 2) * sample_pitch[1]
    y_coords = (y_coords - array_shape[-2] // 2) * sample_pitch[0]

    # Compute the distance from the origin to each point
    radius = np.sqrt(x_coords ** 2 + y_coords ** 2 + focal_length**2)

    # Compute the phase shift based on the distance
    # output dimension: [1,C,array_shape[0],array_shape[1]]
    phase_shift = 2 * math.pi * radius[None,None,:,:] / wavelength

    return phase_shift

def plot_result_1d(results, pytorch=True, figsize=(6,3), dpi=200, type='line'):
    """
    画图函数
    :param results: 数据
    :param pytorch: 是否为pytorch tensor
    :param figsize: 图片大小
    :param dpi: 图片分辨率
    :param type: 图形类型
    """
    # 判断是否为pytorch tensor
    if pytorch:
        results = [result.numpy() for result in results]

    # 确定子图数量
    num_subplots = len(results)

    # 确定子图布局
    if num_subplots == 1:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
        axs = [axs]
    elif num_subplots == 2:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi)
    elif num_subplots == 3:
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=figsize, dpi=dpi)
    elif num_subplots == 4:
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=figsize, dpi=dpi)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(nrows=num_subplots, ncols=1, sharex=True, figsize=figsize, dpi=dpi)
        axs = axs.flatten()

    # 绘制不同类型的图形
    if type == 'stem':
        for i, ax in enumerate(axs):
            ax.stem(results[i])
            ax.set_xlabel('index')
            ax.set_ylabel('value')
    elif type == 'line':
        for i, ax in enumerate(axs):
            ax.plot(results[i])
            ax.set_xlabel('index')
            ax.set_ylabel('value')
    elif type == 'bar':
        for i, ax in enumerate(axs):
            ax.bar(np.arange(len(results[i])), results[i])
            ax.set_xlabel('index')
            ax.set_ylabel('value')

    plt.tight_layout()
    plt.show()

def plot_result(results,
                labels=None,channel_labels=None,
                pytorch=True,
                channel=None,
                figsize=(6,3),dpi=200,
                normalize=False,
                centered_axis=True,
                log_scale=False, db_min=-70,
                x_axis=None,
                type='stem',
                filename=None):
    
    if channel is None:
        for k, z in enumerate(results):
            if z.ndim==2:
                z = z[None,None,:,:]
            elif z.ndim==1:
                z = z[None,None,:,None]
            elif z.ndim==3:
                z = z[None,:,:,:]
            fig = plt.figure(figsize=figsize,dpi=dpi)
            if labels is not None:
                fig.suptitle(labels[k])
            axes = np.array(fig.subplots(1,z.shape[1])).ravel()
            # subplots
            for i, ax in enumerate(axes):
                if channel_labels is not None:
                    ax.set_title(channel_labels[i])
                if pytorch:
                    z_plot = z[0,i,:,:].detach().cpu().numpy()
                else:
                    z_plot = z[0,i,:,:]
                if normalize:
                    z_plot = z_plot/z_plot.sum()
                if min(z_plot.shape[-1],z_plot.shape[-2])==1:
                    # plot 1D
                    N = z_plot.size
                    if centered_axis:
                        x = cfftorder(N)
                    else:
                        x = np.arange(N)
                    if x_axis is not None:
                        if type=='stem':
                            ax.stem(x_axis,z_plot)
                        elif type=='plot':
                            ax.plot(x_axis,z_plot)
                    else:
                        if type=='stem':
                            ax.stem(x,z_plot)
                        elif type=='plot':
                            ax.plot(x,z_plot)
                    if log_scale:
                        ax.set_yscale('log')
                        ax.set_ylim(10**(db_min/10))
                    ax.grid(True)
                else:
                    # plot 2D
                    if log_scale:
                        z_plot = pow2db(z_plot,db_min)
                        im = ax.imshow(z_plot,interpolation='none',aspect='auto')
                        im.set_clim(db_min,0)
                    else:
                        im = ax.imshow(z_plot,interpolation='none',aspect='auto')
                    # im = ax.imshow(z_plot,aspect='auto')
                    plt.colorbar(im,ax=ax)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename + '.png')
            plt.close(fig)
    else:
        fig = plt.figure(figsize=figsize,dpi=dpi)
        axes = np.array(fig.subplots(1,len(results))).ravel()
        # subplots
        for k, (z, ax) in enumerate(zip(results,axes)):
            if z.ndim==2:
                z = z[None,None,:,:]
            elif z.ndim==1:
                z = z[None,None,:,None]
            elif z.ndim==3:
                z = z[None,:,:,:]
            if labels is not None:
                ax.set_title(labels[k])
            if pytorch:
                z_plot = z[0,channel,:,:].detach().cpu().numpy()
            else:
                z_plot = z[0,channel,:,:]
            if normalize:
                z_plot = z_plot/z_plot.sum()
            if min(z_plot.shape[-1],z_plot.shape[-2])==1:
                # plot 1D
                if pytorch:
                    N = z_plot.numel()
                else:
                    # numpy
                    N = z_plot.size
                if centered_axis:
                        x = cfftorder(N)
                else:
                    x = np.arange(N)
                if x_axis is not None:
                    if type=='stem':
                        ax.stem(x_axis,z_plot)
                    elif type=='plot':
                        ax.plot(x_axis,z_plot)
                else:
                    if type=='stem':
                        ax.stem(x,z_plot)
                    elif type=='plot':
                        ax.plot(x,z_plot)
                if log_scale:
                    ax.set_yscale('log')
                    ax.set_ylim(10**(db_min/10))
                ax.grid(True)
            else:
                # plot 2D
                # plot 2D
                if log_scale:
                    z_plot = pow2db(z_plot,db_min)
                    # im = ax.imshow(z_plot,aspect='auto')
                    im = ax.imshow(z_plot,interpolation='none',aspect='auto')
                    im.set_clim(db_min,0)
                else:
                    # im = ax.imshow(z_plot,aspect='auto')
                    im = ax.imshow(z_plot,interpolation='none',aspect='auto')
                plt.colorbar(im,ax=ax)
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename + '.png')
            plt.close(fig)
    

def write_sgd_summary(slm_phase, out_amp, target_amp,loss_value, k,
                      writer=None, path=None, s=1., prefix='test'):
    """tensorboard summary for SGD

    :param slm_phase: Use it if you want to save intermediate phases during optimization.
    :param out_amp: PyTorch Tensor, Field amplitude at the image plane.
    :param target_amp: PyTorch Tensor, Ground Truth target Amplitude.
    :param k: iteration number.
    :param writer: SummaryWriter instance.
    :param path: path to save image files.
    :param s: scale for SGD algorithm.
    :param prefix:
    :return:
    """
    # psnr_value = psnr(target_amp.squeeze().cpu().detach().numpy(), (s * out_amp).squeeze().cpu().detach().numpy())
    # ssim_value = ssim(target_amp.squeeze().cpu().detach().numpy(), (s * out_amp).squeeze().cpu().detach().numpy())

    # s_min = (target_amp * out_amp).mean() / (out_amp**2).mean()
    # psnr_value_min = psnr(target_amp.squeeze().cpu().detach().numpy(), (s_min * out_amp).squeeze().cpu().detach().numpy())
    # ssim_value_min = ssim(target_amp.squeeze().cpu().detach().numpy(), (s_min * out_amp).squeeze().cpu().detach().numpy())

    if writer is not None:
        image_norm = ((s * slm_phase).squeeze(0))/((s * slm_phase).squeeze(0)).max()
        writer.add_image(f'{prefix}_Input/dose', image_norm, k)
        image_norm = ((out_amp).squeeze(0))/((out_amp).squeeze(0)).max()
        writer.add_image(f'{prefix}_Recon/amp', image_norm, k)
        image_norm = ((target_amp).squeeze(0))/((target_amp).squeeze(0)).max()
        writer.add_image(f'{prefix}_Recon/height', image_norm, k)
        writer.add_scalar(f'{prefix}_loss', loss_value, k)
        # writer.add_scalar(f'{prefix}_psnr', psnr_value, k)
        # writer.add_scalar(f'{prefix}_ssim', ssim_value, k)

        # writer.add_scalar(f'{prefix}_psnr/scaled', psnr_value_min, k)
        # writer.add_scalar(f'{prefix}_ssim/scaled', ssim_value_min, k)

        writer.add_scalar(f'{prefix}_scalar', s, k)


def write_gs_summary(slm_field, recon_field, target_amp, k, writer, roi=(880, 1600), prefix='test'):
    """tensorboard summary for GS"""
    slm_phase = slm_field.angle()
    recon_amp, recon_phase = recon_field.abs(), recon_field.angle()
    loss = nn.MSELoss().to(recon_amp.device)

    recon_amp = crop_image(recon_amp, target_shape=roi)
    target_amp = crop_image(target_amp, target_shape=roi)

    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                  / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))

    loss_value = loss(recon_amp, target_amp)
    # psnr_value = psnr(target_amp.squeeze().cpu().detach().numpy(), recon_amp.squeeze().cpu().detach().numpy())
    # ssim_value = ssim(target_amp.squeeze().cpu().detach().numpy(), recon_amp.squeeze().cpu().detach().numpy())

    if writer is not None:
        image_norm = (recon_amp.squeeze(0))/(recon_amp.squeeze(0)).max()
        writer.add_image(f'{prefix}_Recon/amp', image_norm, k)
        writer.add_scalar(f'{prefix}_loss', loss_value, k)
        # writer.add_scalar(f'{prefix}_psnr', psnr_value, k)
        # writer.add_scalar(f'{prefix}_ssim', ssim_value, k)
    return loss_value

def save_tmp_bmp(dose, num=1, save_wrap_res=None,save_pixel_multiply=None):
    path = '.\\checkpoints'
    dose = dose.squeeze().cpu().detach().numpy()
    dose[dose>255] = 255
    dose[dose<0] = 0
    dose = dose.astype(np.uint8)

    if dose.ndim ==1:
        dose = dose[:,None]
        
    # pad using wrapping
    if save_wrap_res is not None:
        dose_save = pad_image(dose, save_wrap_res,
                            pytorch=False,
                            mode='wrap')
    else:
        dose_save = dose

    if save_pixel_multiply is not None:
        # change pixel size
        dose_save = np.kron(dose_save, np.ones((save_pixel_multiply, save_pixel_multiply)))
    cv2.imwrite(os.path.join(path,'iter'+str(num)+'_tmp.bmp'), dose_save.astype(np.uint8))

def save_checkpoints(filename,x,s):
    path = '.\\checkpoints'
    torch.save(x, os.path.join(path,filename+'_x.pt'))
    torch.save(s, os.path.join(path,filename+'_s.pt'))
def load_checkpoints(filename,device=torch.device('cuda')):
    path = '.\\checkpoints'
    x = torch.load(os.path.join(path,filename+'_x.pt')).to(device)
    x = x.clone().detach()
    # x = x.clone().detach().flip(2)
    x = x.requires_grad_(True)
    s = torch.load(os.path.join(path,filename+'_s.pt')).to(device)
    s = s.clone().detach()
    s = s.requires_grad_(True)
    return x,s
def get_psnr_ssim(recon_amp, target_amp, multichannel=False):
    """get PSNR and SSIM metrics"""
    psnrs, ssims = {}, {}

    # amplitude
    psnrs['amp'] = psnr(target_amp, recon_amp)
    ssims['amp'] = ssim(target_amp, recon_amp, multichannel=multichannel)

    # linear
    target_linear = target_amp**2
    recon_linear = recon_amp**2
    psnrs['lin'] = psnr(target_linear, recon_linear)
    ssims['lin'] = ssim(target_linear, recon_linear, multichannel=multichannel)

    # srgb
    target_srgb = srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_linear, 0.0, 1.0))
    psnrs['srgb'] = psnr(target_srgb, recon_srgb)
    ssims['srgb'] = ssim(target_srgb, recon_srgb, multichannel=multichannel)

    return psnrs, ssims


def str2bool(v):
    """ Simple query parser for configArgParse (which doesn't support native bool from cmd)
    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def make_kernel_gaussian(sigma, kernel_size):

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2
    variance = sigma**2

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = ((1 / (2 * math.pi * variance))
                       * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)
                                   / (2 * variance)))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    return gaussian_kernel


def quantized_phase(phasemap):
    """
    just quantize phase into 8bit and return a tensor with the same dtype
    :param phasemap:
    :return:
    """

    # Shift to [0 1]
    phasemap = (phasemap + np.pi) / (2 * np.pi)

    # Convert into integer and take rounding
    phasemap = torch.round(255 * phasemap)

    # Shift to original range
    phasemap = phasemap / 255 * 2 * np.pi - np.pi
    return phasemap

import numpy as np

def generate_gaussian_2d(output_shape, fwhm, center, pitch):
    # output_shape: the shape of the output numpy array
    # fwhm, center, pitch: in real units
    # Calculate the standard deviation from the FWHM and pixel pitch
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Generate 1D grids of coordinates for each axis
    x = cfftorder(output_shape[1]) * pitch[1]
    y = cfftorder(output_shape[0]) * pitch[0]
    
    # Create a 2D grid of coordinates from the 1D grids
    x, y = np.meshgrid(x, y)
    pos = np.dstack((x, y))
    
    # Create a 2D Gaussian function
    center = np.array(center)
    diff = pos - center
    mahalanobis = np.sum(diff**2, axis=2) / (2 * sigma**2)
    gaussian = np.exp(-mahalanobis)
    gaussian = gaussian/np.sqrt((gaussian**2).sum())
    return gaussian

def generate_lorentzian_2d(output_shape, fwhm, center, pitch):
    """
    Generate a 2D Lorentzian distribution.

    Parameters
    ----------
    output_shape : tuple
        The shape of the output numpy array.
    fwhm : float
        Full width at half maximum in real units.
    center : tuple
        The center of the Lorentzian distribution in real units.
    pitch : tuple
        The pixel pitch in real units.

    Returns
    -------
    lorentzian : numpy.ndarray
        The generated 2D Lorentzian distribution.
    """
    # Calculate the gamma parameter from the FWHM
    gamma = fwhm / 2
    
    # Generate 1D grids of coordinates for each axis
    x = cfftorder(output_shape[1]) * pitch[1]
    y = cfftorder(output_shape[0]) * pitch[0]

    # Create a 2D grid of coordinates from the 1D grids
    x, y = np.meshgrid(x, y)
    pos = np.dstack((x, y))

    # Create a 2D Lorentzian function
    center = np.array(center)
    diff = pos - center
    squared_sum = np.sum(diff**2, axis=2)
    lorentzian = (1 / np.pi) * (gamma / (squared_sum + gamma**2))
    lorentzian = lorentzian / np.sqrt((lorentzian**2).sum())
    return lorentzian


def generate_gaussian_array(output_shape, 
                            fwhm, 
                            distort_corr,
                            theta_max,
                            num_x, num_y, 
                            margin, 
                            dim=2):
    # output_shape: the shape of the output numpy array
    # fwhm; pitch: in unit of pixels
    # margin: in unit of pixels
    # the array with a size of [num_y,num_x]

    
    if distort_corr:
        # correction for k-space distortion
        if dim == 1:
            theta_m = np.arcsin(np.sin(theta_max) * 
                                     ((output_shape[0]-margin*2)/
                                     (output_shape[0])))
            # range of s: [-1,1]
            s = 2 * cfftfreq(num_x) 
            # range of theta_x: [-theta_m,theta_m]
            theta_x = s * theta_m 
            # range of normalized f_x: 0.5*[-sin(theta_m)/sin(theta_max),sin(theta_m)/sin(theta_max)]
            # where normalized f_x := f_x/f_boundary/2 = (f_x/f_0) / (f_boundary/f_0) /2
            # and note that f_boundary/f_0 = sin(theta_max)
            f_x = np.sin(theta_x) / np.sin(theta_max) / 2 
            # map from normalized spatial frequency to index (fft orders)
            # map from [-2,-1,0,1]/N (N=4) to [0,1,2,3]
            # map from [-1,0,1]/N (N=3) to [0,1,2]
            centers = (f_x * output_shape[0]) + (output_shape[0]//2)
        else:
            theta_m = np.arcsin(np.sin(theta_max) * 
                                     ((output_shape[0]-margin*2)/
                                     (output_shape[0])))
            # range of s: [-1,1]
            s_x = 2 * cfftfreq(num_x) 
            s_y = 2 * cfftfreq(num_y) 
            # range of theta: [-theta_m,theta_m]
            theta_x = s_x * theta_m 
            theta_y = s_y * theta_m 
            Theta_x, Theta_y = np.meshgrid(theta_x, theta_y)
            T = 1/np.sqrt((np.tan(Theta_x)**2 + np.tan(Theta_y)**2 + 1))
            # normalized f_x, f_y (within and less than [-0.5,0.5])
            # where normalized f_x := f_x/f_boundary/2= (f_x/f_0) / (f_boundary/f_0) /2
            # using relation:
            #   tan(theta_x) = fx/fz; tan(theta_y) = fy/fz
            #   => fz/f0 = 1/sqrt(1+T); 
            #      fx/f0 = fz/f0 tan(theta_x); 
            #      fy/f0 = fz/f0 tan(theta_y)
            # where T = tan(theta_x)^2 + tan(theta_y)^2
            # also, note that f_boundary/f_0 = sin(theta_max)
            f_x = np.tan(Theta_x) * T /np.sin(theta_max) /2
            f_y = np.tan(Theta_y) * T /np.sin(theta_max) /2
            # map from normalized spatial frequency to index (fft orders)
            centers_x =  (f_x * output_shape[0]) + (output_shape[0]//2)
            centers_y =  (f_y * output_shape[0]) + (output_shape[0]//2)
            
    else:
        if dim == 1:
            # Generate 1D grids of coordinates for each axis
            x = np.linspace(0, output_shape[0]-1, output_shape[0])
            centers = np.linspace(margin, output_shape[0]-margin, num_x)
        else:
            x = np.linspace(0, output_shape[1]-1, output_shape[1])
            y = np.linspace(0, output_shape[0]-1, output_shape[0])
            X, Y = np.meshgrid(x, y)
            cx = np.linspace(margin, output_shape[1]-margin, num_x)
            cy = np.linspace(margin, output_shape[0]-margin, num_y)
            centers_x, centers_y = np.meshgrid(cx, cy)

    output = np.zeros(output_shape)

    if dim == 1:
        for i in range(num_x):
            if fwhm==None:
                output[int(round(centers[i]))] = 1.0
            else:
                sigma = math.ceil(fwhm / (2 * np.sqrt(2 * np.log(2))))
                gaussian = np.exp(-(x-centers[i])**2/(2*(sigma)**2))
                output += gaussian
    else:
        for i in range(num_x):
            for j in range(num_y):
                if fwhm==None:
                    output[int(round(centers_y[i][j])),int(round(centers_x[i][j]))] = 1.0
                    
                else:
                    sigma = math.ceil(fwhm / (2 * np.sqrt(2 * np.log(2))))
                    gaussian = np.exp(-(X-centers_x[i][j])**2/(2*(sigma)**2) -
                                       (Y-centers_y[i][j])**2/(2*(sigma)**2))
                    output += gaussian

    output = output/np.sqrt((output**2).sum())

    return output

def normalize(input,method='sum',pytorch=True):
    if input.ndim>1:
        dim = (-2,-1)
        sz = input.shape[-2]*input.shape[-1]
    else:
        dim = (-1)
        sz = input.shape[-1]
    if pytorch:
        if method=='sum':
            return input/input.sum(dim=dim,keepdim=True) * (sz)
        elif method=='sqrt':
            return input/((input**2).sum(dim=dim,keepdim=True)).sqrt() * math.sqrt(sz)
        elif method=='max':
            return input/input.max()
        elif method=='None':
            return input
    else:
        if method=='sum':
            return input/input.sum(axis=dim,keepdims=True) * (sz)
        elif method=='sqrt':
            return input/np.sqrt((input**2).sum(axis=dim,keepdims=True)) * math.sqrt(sz)
        elif method=='max':
            return input/input.max()
        elif method=='None':
            return input


def extract_subpatterns(img, threshold_value=0.1):
    """
    Extracts subpatterns from an image and rearranges them in a new picture for better visualization.

    Parameters:
        img (numpy.ndarray): The input image.
        pattern_size (tuple): The size of each subpattern to extract, given as a tuple of (height, width).
        threshold_value: Set the relative threshold value to separate the subpatterns from the background
        padding_size (tuple): The size of the padding to add around each subpattern, given as a tuple of (height, width).

    Returns:
        numpy.ndarray: The output image containing the rearranged subpatterns.
    """
    # for cv2 findContours function
    img = img/img.max()*255.0
    img = img.astype(np.uint8)
    # Create a binary mask of the picture using the threshold value
    mask = np.zeros_like(img)
    mask[img > threshold_value*img.max()] = 1

    # Find the contours of the subpatterns using the binary mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract the rectangular regions containing the subpatterns and store them in a list
    subpatterns = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        subpatterns.append(img[y:y+h, x:x+w]) 
    
    num_subpatterns = len(subpatterns)
    rows = int(math.ceil(math.sqrt(num_subpatterns)))
    cols = int(math.ceil(num_subpatterns / float(rows)))
    # determine size of each tile
    large_pic_shape = subpatterns[0].shape
    tile_width = int(large_pic_shape[1] / cols)
    tile_height = int(large_pic_shape[0] / rows)
    # create large picture
    large_pic = np.zeros((large_pic_shape[0], large_pic_shape[1]), dtype=np.uint8)
    # loop through subpatterns and insert into large picture
    for i, subpattern in enumerate(subpatterns):
        row_idx = int(i / cols)
        col_idx = i % cols
        start_y = row_idx * tile_height
        end_y = start_y + tile_height
        start_x = col_idx * tile_width
        end_x = start_x + tile_width
        resized_subpattern = cv2.resize(subpattern, (tile_width, tile_height))
        large_pic[start_y:end_y, start_x:end_x] = resized_subpattern
    return large_pic