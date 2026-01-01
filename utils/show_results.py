import numpy as np
import utils.utils as utils

def show_results(opt, recon_field,
            target_field=None, 
            dpi=400,log_scale=True,db_min=-20,
            pytorch=True):
    '''
    input field shape: [B,C,H,W]
    '''
    if pytorch:
        recon_field = recon_field.cpu().detach().numpy()
        if target_field != None:
            target_field = target_field.cpu().detach().numpy()
    batch_num = recon_field.shape[0]
    channel_num = recon_field.shape[1]

    ptype = opt.pattern_type
    if ptype == 'grating_array':
        eff_rel = np.zeros(batch_num)
        for i in range(batch_num):
            eff_rel[i] = diffraction_efficiency(recon_field[i],
                                                target_field[i],
                                                plot=False,pytorch=False)
        utils.plot_result([eff_rel], 
                ['Rel. Eff.'],
                figsize=(11,3),centered_axis=False,
                pytorch=False,dpi=dpi,normalize=True,log_scale=log_scale,db_min=db_min)
    # elif ptype == 'grating':
    elif ptype =='pic':
        for i in range(batch_num):
            utils.plot_result([recon_field[i],target_field[i]], 
                    ['Recon.','Target'],
                    figsize=(11,3),centered_axis=False,
                    pytorch=False,dpi=dpi,normalize=True,log_scale=log_scale,db_min=db_min)

def show_batch_results(opt, recon_field,
                            target_field=None, 
                            dpi=400,log_scale=True,db_min=-20,
                            pytorch=True):
    '''
    input field shape: [1,C,H,W]
    '''
    if pytorch:
        recon_field = recon_field.cpu().detach().numpy()
        if target_field != None:
            target_field = target_field.cpu().detach().numpy()
    channel_num = recon_field.shape[1]
    ptype = opt.pattern_type
    if ptype == 'grating_array':
        recon_amp = np.abs(recon_field)
        diffraction_efficiency(recon_field,target_field,
                                plot=True,pytorch=False)
    elif ptype =='pic':
        recon_amp = np.abs(recon_field)
        utils.plot_result([recon_field,target_field], 
                    ['Recon.','Target'],
                    figsize=(11,3),centered_axis=False,
                    pytorch=False,dpi=dpi,normalize=True,log_scale=log_scale,db_min=db_min)


def diffraction_efficiency(recon_field,
                            target_field,
                            pytorch=True,
                            plot=False):
    if pytorch:
        recon_amp = np.abs(recon_field.cpu().detach().numpy())
        target_amp = np.abs(target_field.cpu().detach().numpy())
    else:
        recon_amp = np.abs(recon_field)
        target_amp = np.abs(target_field)

    index = int((target_amp.squeeze().nonzero())[0])
    diff_pattern = recon_amp.squeeze()
    eff_rel = diff_pattern[index]**2/(diff_pattern**2).sum()
    if plot:
        utils.plot_result([recon_amp**2],
                [' Rel. DE '+str(np.round(eff_rel,3))],
                pytorch=False,normalize=True,semilogy=True)
    return eff_rel