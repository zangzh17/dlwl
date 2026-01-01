import os
import numpy as np
import math
import torch
from PIL import Image
from copy import deepcopy
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from utils.utils import crop_image, pad_image
import utils.utils as utils

def pattern_selector(opt,
                     plot=True,
                     device=torch.device('cpu'),
                     dtype=torch.float32):
    if opt.pattern_type == 'splitter':
        im = splitter(opt.pattern_params.dim,
                    opt.pattern_params.orders,opt.pattern_params.order_shift,
                    opt.pattern_params.roi_order_num, 
                    opt.pattern_params.tot_order_num, 
                    device,dtype)
        opt.target_res = (im.shape[-2],im.shape[-1])
        opt.slm_res = opt.target_res 
        opt.output_res = opt.target_res 
        order_str = '_'.join([str(i) for i in opt.pattern_params.orders])+'_'+'_'.join([str(i) for i in opt.pattern_params.order_shift])
        if opt.pattern_params.load:
            opt.load_checkpoint_file = 'spl_'+str(opt.pattern_params.dim)+'d_'+order_str+'_'+str(opt.pattern_params.roi_order_num)+'_'+str(opt.pattern_params.tot_order_num)
        if opt.pattern_params.save:
            opt.save_checkpoint_file = 'spl_'+str(opt.pattern_params.dim)+'d_'+order_str+'_'+str(opt.pattern_params.roi_order_num)+'_'+str(opt.pattern_params.tot_order_num)
        return [opt], [im]
    elif opt.pattern_type == 'fresnel_lens':
        im = fresnel_lens(opt,device,dtype)
        opt.slm_res = opt.target_res 
        opt.output_res = opt.target_res 
        im = (im*opt.normalize_ratio)
        if plot:
            utils.plot_result([im**2],['Targets'],dpi=700,log_scale=True)
        return [opt], [im]
    elif opt.pattern_type == 'splitter_array':
        im, wavelengths = splitter_array(opt.pattern_params.dim,
                                            opt.pattern_params.N_group,
                                            opt.pattern_params.N_pattern,
                                            opt.pattern_params.wavelength_range,
                                            opt.pattern_params.orders_start,
                                            opt.pattern_params.orders_step,
                                            opt.pattern_params.roi_order_num,
                                            opt.pattern_params.tot_order_num,
                                            device,dtype)
        opt_list =  [deepcopy(opt) for i in range(len(im))]
        for i in range(len(im)):
            opt_list[i].target_res = (im[i].shape[-2],im[i].shape[-1])
            opt_list[i].slm_res = opt_list[i].target_res
            opt_list[i].output_res = opt_list[i].target_res 
            opt_list[i].wavelength = [wavelengths[i]]
            pattern_idx = i%opt.pattern_params.N_pattern
            order_str = '_'.join([str(j) for j in opt.pattern_params.orders_start])+'_'+'_'.join([str(j) for j in opt.pattern_params.orders_step[pattern_idx]])        
            if opt.pattern_params.load:
                opt_list[i].load_checkpoint_file = 'spl_'+str(opt.pattern_params.dim)+'d_'+order_str+'_'+str(opt.pattern_params.roi_order_num)+'_'+str(opt.pattern_params.tot_order_num)
            if opt.pattern_params.save:
                opt_list[i].save_checkpoint_file = 'spl_'+str(opt.pattern_params.dim)+'d_'+order_str+'_'+str(opt.pattern_params.roi_order_num)+'_'+str(opt.pattern_params.tot_order_num)
        return opt_list, im    
    elif opt.pattern_type == 'splitter_multi':
        im = splitter_multi(opt.pattern_params.orders,
                        opt.pattern_params.roi_order_num, 
                        opt.pattern_params.tot_order_num, 
                        device,dtype)
        opt.target_res = (im.shape[-2],im.shape[-1])
        opt.slm_res = opt.target_res 
        opt.output_res = opt.target_res 
        if opt.pattern_params.load:
            opt.load_checkpoint_file = 'spl_multi_'+opt.pattern_params.label
        if opt.pattern_params.save:
            opt.save_checkpoint_file = 'spl_multi_'+opt.pattern_params.label
        return [opt], [im]
    elif opt.pattern_type == 'grating':
        im = blazed_grating(opt.pattern_params.period, 
                            opt.feature_size[0],
                            opt.pattern_params.order,
                            device,dtype)
        opt.target_res = (im.shape[0],1)
        opt.slm_res = opt.target_res 
        opt.output_res = opt.target_res 
        if opt.pattern_params.load:
            opt.load_checkpoint_file = 'grating_'+str(im.shape[0])+'_'+str(opt.pattern_params.order)
        if opt.pattern_params.save:
            opt.save_checkpoint_file = 'grating_'+str(im.shape[0])+'_'+str(opt.pattern_params.order)
        return [opt], [im[None,None,:,None]]
    elif opt.pattern_type == 'grating_array':
        im, wavelengths, orders, roi_res = grating_array(opt.pattern_params.period_range,
                            opt.pattern_params.orders_range,
                            opt.pattern_params.roi_length,
                            opt.pattern_params.wavelength_range,
                            opt.pattern_params.N_group,
                            opt.feature_size[0],
                            device,dtype)
        opt_list =  [deepcopy(opt) for i in range(len(im))]
        for i in range(len(im)):
            opt_list[i].target_res = (im[i].numel(),1)
            opt_list[i].slm_res = opt_list[i].target_res
            opt_list[i].output_res = opt_list[i].target_res 
            opt_list[i].roi_res = (roi_res[i],1)
            opt_list[i].wavelength = [wavelengths[i]]
            if opt.pattern_params.load:
                opt_list[i].load_checkpoint_file = 'grating_'+str(im[i].numel())+'_'+str(orders[i])
            if opt.pattern_params.save:
                opt_list[i].save_checkpoint_file = 'grating_'+str(im[i].numel())+'_'+str(orders[i])
        if plot:
            thetas = np.zeros(len(im))
            sz = np.zeros(len(im))
            for i,m in enumerate(im):
                order_ind = m.ravel().nonzero()[0].detach().cpu().numpy()
                sz[i] = im[i].numel()
                order = utils.cfftorder(int(sz[i]))[order_ind]
                thetas[i] = np.rad2deg(np.arcsin(order * opt_list[i].wavelength[0]/(sz[i]*opt_list[i].feature_size[0])))
            utils.plot_result([sz, thetas],['size','theta'],dpi=600,
                                pytorch=False,
                                centered_axis=False)
        return opt_list, im
    elif opt.pattern_type == 'pic':
        im = load_image(opt,device, dtype)
        if plot:
            utils.plot_result([im**2],['Intensity'],dpi=700,log_scale=True)
            utils.plot_result([im],['Amp.'],dpi=700,log_scale=False)
        return [opt], [im]
    elif opt.pattern_type == 'pt':
        im = load_pt(opt.pattern_params.data_path, opt.pattern_params.target_filename,
                    opt.target_res,opt.slm_res,opt.roi_res,
                    device, dtype)
        im = (im*opt.normalize_ratio).repeat(1,opt.wavelength.size,1,1)
        batch_num = im.shape[0]
        if batch_num>1:
            plot_amp = utils.tile_batch(im,opt.plot_layout,pytorch=True)
        else:
            plot_amp = im
        utils.plot_result([plot_amp**2],['Targets'],dpi=700)
        return [opt]*batch_num, im.split(1,dim=0)
    elif opt.pattern_type == 'fza':
        im = zone_plate(opt.pattern_params.r1 , opt.target_res, opt.feature_size,
                        device, dtype)
        im = (im*opt.normalize_ratio).repeat(1,opt.wavelength.size,1,1)
        if plot:
            utils.plot_result([im**2],['Targets'],dpi=700)
        return [opt], [im]
    elif opt.pattern_type == 'dot_projector':
        im = dot_projector(opt,device,dtype)
        if plot:
            utils.plot_result([im**2],['Targets'],dpi=700)
        return [opt], [im]
    # elif opt.pattern_type == 'height':
    #     im
    #     return [opt], [im]
    

def fresnel_lens(opt,device= torch.device('cuda'),
                 dtype=torch.float32):
    num_channel = max([opt.wavelength.size,opt.prop_dist.size])
    # diffraction limited spot size in fwhm
    if opt.prop_model == 'SFR':
        Y = opt.output_size[0]
        X = opt.output_size[1]
    else:
        Y = opt.slm_res[0]*opt.feature_size[0]
        X = opt.slm_res[1]*opt.feature_size[1]
    dy = float(opt.wavelength.min() * opt.pattern_params.focal_length/Y)
    dx = float(opt.wavelength.min() * opt.pattern_params.focal_length/X)
    fwhm = 1.025 * min([dx,dy]) * opt.pattern_params.relative_fwhm

    im = np.zeros((1,num_channel,opt.target_res[0],opt.target_res[1]))
    for i in range(num_channel):
        # im[0,i,:,:] = utils.generate_gaussian_2d(opt.target_res, 
        #                                         fwhm, 
        #                                         opt.pattern_params.center[i], 
        #                                         opt.feature_size)
        im[0,i,:,:] = utils.generate_lorentzian_2d(opt.target_res, 
                                                fwhm, 
                                                opt.pattern_params.center[0], 
                                                opt.feature_size)
    im = torch.from_numpy(im).to(device=device,dtype=dtype)   
    im = utils.normalize(im,method='sqrt')
    return im

def dot_projector(opt,device= torch.device('cuda'),
                 dtype=torch.float32):

    im = np.zeros((1,1,opt.target_res[0],opt.target_res[1]))
    # calculate maximum deflection angle along a single axis
    theta_max = math.asin(float(opt.target_res[0]/opt.slm_res[0] *
                                opt.wavelength.min()/2/opt.feature_size[0]))
    im[0,0,:,:] = utils.generate_gaussian_array(opt.target_res, 
                                                opt.pattern_params.fwhm, 
                                                opt.pattern_params.distort_corr,
                                                theta_max,
                                                opt.pattern_params.num_x, 
                                                opt.pattern_params.num_y,
                                                opt.pattern_params.margin, 
                                                opt.pattern_params.dim)
    im = torch.from_numpy(im).to(device=device,dtype=dtype)   
    im = utils.normalize(im,method='sqrt')
    return im

def splitter(dim = 1,
            orders=(0,3,6),order_shift=(0,0),
            roi_order_num=10, 
            tot_order_num=40,
            device= torch.device('cuda'),dtype=torch.float32):
    """generate spot array pattern
        dim: 1 for 1D, 2 for 2D spots array
        orders: indicies of working orders of spots array 
        order_num: total number of orders
        note that 
        1. the total FoV is decided by feature_size
        2. the roi FoV is roi_order_num/tot_order_num of the total FoV
        i.e. effective feature_size = feature_size * tot_order_num/roi_order_num
    """

    if dim ==1:
        # apply order shift
        orders = [(ord+int(order_shift[0]))%roi_order_num for ord in orders]
        im = np.zeros(roi_order_num)
        I = 1.0/np.sqrt(len(orders))
        for i in orders:
            im[i] = I
        im = utils.pad_image(im, tot_order_num, pytorch=False)
        im = torch.from_numpy(im).to(device=device,dtype=dtype)
        im = im[None,None,:,None]
    elif dim==2:
        # apply order shift
        orders_y = [(ord+int(order_shift[0]))%roi_order_num for ord in orders]
        orders_x = [(ord+int(order_shift[1]))%roi_order_num for ord in orders]
        im = np.zeros((1,1,roi_order_num,roi_order_num))
        I = 1.0/np.sqrt(len(orders))
        for ord_y in orders_y:
            for ord_x in orders_x:
                im[0,0,ord_y,ord_x] = I
        im = utils.pad_image(im, (tot_order_num,tot_order_num), pytorch=False)
        im = torch.from_numpy(im).to(device=device,dtype=dtype)
    else:
        raise Exception("splitter dim should be 1 or 2")
    
    im = utils.normalize(im,method='sqrt')
    return im

def splitter_array(dim, N_group, N_pattern, wavelength_range,
                    orders_start, orders_step,
                    roi_order_num, tot_order_num,
                    device= torch.device('cuda'),
                    dtype=torch.float32):
    wavelengths = np.linspace(wavelength_range[0],
                              wavelength_range[1],
                              N_pattern*N_group)
    im = []
    
    for k in range(N_group):
        for i in range(N_pattern):
            # gen splitter unit
            target_amp = splitter(dim, orders_start, orders_step[i],
                            roi_order_num, tot_order_num,
                            device=device,dtype=dtype)
            target_amp = utils.normalize(target_amp,method='sqrt')
            im.append(target_amp)

    return im, wavelengths

def splitter_multi(orders,
                    roi_order_num=10, 
                    tot_order_num=40,
                    device= torch.device('cuda'),dtype=torch.float32):
    """generate spot array pattern with multiple working wavelengths
    """
    im = np.zeros((1,len(orders),roi_order_num,roi_order_num))
    for i,order in enumerate(orders):
        for ord_y in order:
            for ord_x in order:
                im[0,i,ord_y,ord_x] = 1.0
    im = utils.pad_image(im, (tot_order_num,tot_order_num), pytorch=False)
    im = torch.from_numpy(im).to(device=device,dtype=dtype)
    im = utils.normalize(im,method='sqrt')
    return im

def blazed_grating(period, 
                    feature_size,
                    order=1, 
                    device= torch.device('cuda'),
                    dtype=torch.float32):
    """generate blazed grating pattern
    """
    N = int(np.round(np.abs(period)/feature_size))
    im = np.zeros(N)
    index = utils.cfftorder(N) == order
    im[index] = 1.0
    im = torch.from_numpy(im).to(device=device,dtype=dtype)
    im = utils.normalize(im,method='sqrt')
    return im

def grating_array(period_range,
                  orders_range,
                  roi_length,
                  wavelength_range,
                  N_group,
                  pixel_size,
                  device= torch.device('cuda'),dtype=torch.float32):
    N_theta = len(period_range)*2 + 1
    wavelengths = np.linspace(wavelength_range[0],
                              wavelength_range[1],
                              N_theta*N_group).reshape((-1,N_theta))
    im = []
    orders = np.zeros_like(wavelengths)
    roi_res = np.zeros_like(wavelengths)
    period_range = np.array(period_range)
    orders_range = np.array(orders_range)
    roi_length = np.array(roi_length)
    
    for i in range(N_group):
        # periods_px = np.round(np.array(period_range)/pixel_size * max_order).astype(int)
        # factor = np.gcd(periods_px,max_order).astype(int)
        # periods_px = periods_px/factor
        # orders[i] = np.concatenate((max_order/factor,np.array([0]),-max_order/factor[-1::-1]))
        # periods[i] = np.concatenate((periods_px*pixel_size,np.array([pixel_size*2]),periods_px[-1::-1]*pixel_size))
        orders[i] = np.concatenate((orders_range ,np.array([0]),-orders_range[-1::-1]))
        roi_res[i] = np.concatenate((roi_length ,np.array([2]),roi_length[-1::-1]))
        periods = np.concatenate((period_range,np.array([pixel_size*2]),period_range[-1::-1]))
        for p,m in zip(periods,orders[i]):
            target_amp = blazed_grating(p, 
                            pixel_size,order=m, 
                            device=device, dtype=dtype)
            im.append(target_amp[None,None,:,None])

    return im, wavelengths.flatten(), orders.astype(int).flatten(), roi_res.astype(int).flatten()


def load_image(opt,device= torch.device('cuda'), dtype=torch.float32):
    # for image:
    # 0. load, tuning
    # 1. resize to target_res
    # 2. zero padding to roi_res
    # Define list of filenames
    if isinstance(opt.pattern_params.target_filename, str):
        filenames = [opt.pattern_params.target_filename]
    else:
        filenames = opt.pattern_params.target_filename
    # Load first image and convert to grayscale
    im = Image.open(os.path.join(opt.pattern_params.data_path, filenames[0]))
    if opt.pattern_params.rgb:
        im = im.convert('RGB')
    else:
        im = im.convert('L')
    # convert to tensor
    to_tensor = transforms.ToTensor()
    target = to_tensor(im).unsqueeze(0).to(device=device,dtype=dtype)
    # Loop over remaining filenames and concatenate grayscale images along channel dimension
    for filename in filenames[1:]:
        # Load image and convert to grayscale
        im = Image.open(os.path.join(opt.pattern_params.data_path, filename))
        imGray = im.convert('L')
        # Convert grayscale image to tensor and concatenate with previous images
        target = torch.cat((target, to_tensor(imGray).unsqueeze(0).to(device=device,dtype=dtype)), dim=1)

    if opt.pattern_params.reverse:
        for b in range(target.shape[0]):
            for c in range(target.shape[1]):
                max_val = torch.max(target[b, c, :, :])
                target[b, c, :, :] = max_val - target[b, c, :, :]
    if target.shape[1] == 1 or target.shape[1] == 3:
        target = TF.adjust_contrast(target,opt.pattern_params.contrast_factor)

    # linearize intensity
    target = utils.srgb_gamma2lin(target,pytorch=True)

    # resize
    target = F.interpolate(target, opt.target_res, mode='bilinear')
    
    # binarize
    if opt.pattern_params.binarize:
        target[target<target.max()/2]=0
        target[target>target.max()/2]=target.max()
    # energy normalize to 1
    target = target/torch.sum(target,dim=(-2,-1),keepdim=True)
    # sqrt to get amp
    target = target.sqrt()
    if target.shape[1]>1:
        # scaling according to wavelength
        if opt.pattern_params.scale_by_wavelength:
            wavelength =  list(opt.wavelength.squeeze())
            scale_factors = [wavelength[1]/w for w in wavelength]
            for i in range(target.shape[1]): # loop through channels
                if scale_factors[i] < 1.0:
                    # pad with zeros and then scale down
                    padded_size = [int(s/scale_factors[i]) for s in target.shape[-2:]] # compute the padding width
                    scaled_tensor = pad_image(target[:,i,:,:].unsqueeze(dim=1), padded_size) # pad along H and W dim
                    scaled_tensor = F.interpolate(scaled_tensor, opt.target_res, mode='bilinear') # scale down
                    target[:,i,:,:] = scaled_tensor.squeeze(dim=1)
                elif scale_factors[i] > 1.0:
                    # scale up and then crop
                    scaled_tensor = F.interpolate(target[:,i,:,:].unsqueeze(dim=1), scale_factor=scale_factors[i], mode='bilinear') # scale up
                    target[:,i,:,:] = crop_image(scaled_tensor, opt.target_res).squeeze(dim=1) # crop back
    else:
        target =  target.repeat(1,opt.wavelength.size,1,1)
    # padding & normalize
    target = pad_image(target, opt.roi_res)
    return utils.normalize(target,method='sqrt')

def load_pt(data_path, pt_filename,
                target_res,slm_res,roi_res,
                device= torch.device('cuda'), dtype=torch.float32):
    im = torch.load(os.path.join(data_path,pt_filename))
    target_intensity = im.unsqueeze(1).to(dtype=dtype)
    # resize & padding
    target_intensity = F.interpolate(target_intensity, target_res, mode='bicubic')
    target_intensity = pad_image(target_intensity,roi_res)
    target_amp = target_intensity.sqrt()
    return utils.normalize(target_amp.to(device=device,dtype=dtype),method='sqrt')

def zone_plate(r1,target_res,feature_size,
               device= torch.device('cuda'), dtype=torch.float32):
    num_y, num_x = target_res[0], target_res[1]
    dy, dx = feature_size[0], feature_size[1]
    y = np.fft.fftshift(np.fft.fftfreq(num_y) * num_y) *dy 
    x = np.fft.fftshift(np.fft.fftfreq(num_x) * num_x) *dx
    x_grid, y_grid = np.meshgrid(x, y)
    r2 = x_grid**2 + y_grid**2
    im = 0.5 + 0.5*np.cos(np.pi*r2/(r1**2))
    im = torch.from_numpy(im).to(device=device,dtype=dtype)
    # normalize to 1
    target_intensity = im/torch.sum(im,dim=(-2,-1),keepdim=True)
    target_amp = target_intensity.sqrt()
    return utils.normalize(target_amp.to(device=device,dtype=dtype),method='sqrt')
