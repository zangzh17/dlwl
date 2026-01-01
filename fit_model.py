import numpy as np
import matplotlib.pyplot as plt
import pwlf
import os
import re
from scipy.io import loadmat
import torch
import torch.nn as nn


class fit_model:
    """ A module for GT/LP fitting
    using loaded polyfit
    """
    def __init__(self,
                 gt_file,
                 lp_file,
                 lp_ratio = 1.0,
                 show = True,
                 gt_path=r'C:\Users\zzh\MATLAB Drive\DLW-fab\gt_fit_files',
                 lp_path=r'C:\Users\zzh\MATLAB Drive\DLW-fab\lp_fit_files',
                 device=torch.device('cuda'),
                 dtype=torch.float32):
        self.dev = device
        self.dtype = dtype
        self.lp_ratio = lp_ratio
        
        # load data
        gt = loadmat(os.path.join(gt_path,gt_file))
        lp = loadmat(os.path.join(lp_path,lp_file))
        self.gt_x = torch.tensor(gt['gt_model']['dose_x'][0][0][0],device=device,dtype=dtype)
        self.gt_y = torch.tensor(gt['gt_model']['dose_y'][0][0][0],device=device,dtype=dtype)
        self.gt_coeff = torch.tensor(gt['gt_model']['dose_coeff'][0][0][0],device=device,dtype=dtype)
        self.gt_inv_x = torch.tensor(gt['gt_model']['gt_x'][0][0][0],device=device,dtype=dtype)
        self.gt_inv_y = torch.tensor(gt['gt_model']['gt_y'][0][0][0],device=device,dtype=dtype)
        self.gt_inv_coeff = torch.tensor(gt['gt_model']['gt_coeff'][0][0][0],device=device,dtype=dtype)
        self.gt_x.requires_grad = False
        self.gt_y.requires_grad = False
        self.gt_coeff.requires_grad = False
        self.gt_inv_x.requires_grad = False
        self.gt_inv_y.requires_grad = False
        self.gt_inv_coeff.requires_grad = False

        self.gt_mid = float((self.gt_x[0]+self.gt_x[1])/2)
        
        self.lp_f0 = torch.tensor(lp['lp_model']['f_max'][0][0][0],device=device,dtype=dtype)
        self.lp_coeff = torch.tensor(lp['lp_model']['mtf_coeff'][0][0][0],device=device,dtype=dtype)
        self.lp_f0.requires_grad = False
        self.lp_coeff.requires_grad = False
        self.lp_f0_np = lp['lp_model']['f_max'][0][0][0]
        self.lp_type = lp['lp_model']['fit_type'][0][0][0]
        self.lp_type = re.search(r'([a-zA-Z]+)',self.lp_type)[0]
        self.lp_coeff_np = lp['lp_model']['mtf_coeff'][0][0][0]
        # show
        if show:
            self.show()
    
    def show(self):
        print('LP ratio: ', self.lp_ratio)
        x1Hat = np.linspace(0, 255, num=1000)
        x2Hat = np.linspace(0, 1, num=1000)
        y1Hat_torch = self.gt_torch(torch.tensor(x1Hat,device=self.dev,dtype=self.dtype)).cpu().numpy()
        y2Hat_torch = self.lp_torch(torch.tensor(x2Hat,device=self.dev,dtype=self.dtype)).cpu().numpy()
        x3Hat = torch.linspace(y1Hat_torch.min(),y1Hat_torch.max(),steps=1000,device=self.dev,dtype=self.dtype)
        y3Hat_torch = self.gt_inv_torch(x3Hat).cpu().numpy()
        x3Hat = x3Hat.cpu().numpy()
        plt.figure(figsize=(8, 8), dpi=100)
        plt.subplot(3,1,1)
        plt.plot(x1Hat, y1Hat_torch, '-')
        plt.title('GT')
        plt.subplot(3,1,2)
        plt.plot(x2Hat, y2Hat_torch, '-')
        plt.title('LP')
        plt.subplot(3,1,3)
        plt.plot(x3Hat, y3Hat_torch, '-')
        plt.title('GT inv.')
        plt.tight_layout()
        plt.show() 
    def gt_torch(self,x):
        return self.polyval(torch.clamp(x.ravel(),
                                        self.gt_x[0],
                                        self.gt_x[1]),
                            self.gt_coeff).reshape(x.shape)
    def gt_inv_torch(self,x):
        return self.polyval(torch.clamp(x.ravel(),
                                        self.gt_inv_x[0],
                                        self.gt_inv_x[1]),
                            self.gt_inv_coeff).reshape(x.shape)
    def lp(self,x):
        return np.where(x.ravel()>self.lp_f0_np[0],
                        0,
                        self.polyval(np.clip(x.ravel(),
                                        0.0,self.lp_f0_np[0]),
                                    self.lp_coeff_np, 
                                    ftype=self.lp_type,
                                    pytorch=False) * self.lp_ratio).reshape(x.shape)
    def lp_torch(self,x):
        return torch.where(x.ravel()>self.lp_f0[0],
                            0.0, 
                            self.polyval(torch.clamp(x.ravel(),
                                                     0,
                                                     self.lp_f0[0]),
                                         self.lp_coeff,
                                         ftype=self.lp_type)).reshape(x.shape) * self.lp_ratio
        
        # return self.polyval(torch.clamp(x.ravel(),0,self.lp_f0[0]),
        #                     self.lp_coeff,ftype=self.lp_type).reshape(x.shape) * self.lp_ratio
    
    def polyval(self,x,coeffs,ftype='poly',pytorch=True):
        '''
        coeffs: array/tensor([p1,p2,p3,p4]
        Eval: coeffs[n-1] + coeffs[n-2] * x + ... + coeffs[0] * x**(n-1)
        evaluated using Horner's method:
        p(x) = coeffs[n-1] + x * (coeffs[n-2] + ... + x * (coeffs[1] + x * coeffs[0]))
        '''
        if ftype=='poly':
            curVal = coeffs[0]
            for coeff in coeffs[1:]:
                curVal = coeff + x * curVal
        elif ftype=='exp':
            curVal = 0.0
            if pytorch:
                for a,b in zip(coeffs[::2],coeffs[1::2]):
                    curVal = curVal + a*torch.exp(b*x)
            else:
                for a,b in zip(coeffs[::2],coeffs[1::2]):
                    curVal = curVal + a*np.exp(b*x)
        elif ftype=='power':
            if pytorch:
                curVal = coeffs[0]*torch.pow(x,coeffs[1]) +coeffs[2]
            else:
                curVal = coeffs[0]*np.power(x,coeffs[1]) +coeffs[2]                 
        return curVal
    

class fit_pwl_model:
    """ A module for GT/LP fitting
    using piecewise linear fitting
    """
    def __init__(self,
                 gt_file,
                 lp_file,
                 lp_ratio = 1.0,
                 show = True,
                 gt_seg_num=4, lp_seg_num=2,
                 gt_pop = 20, lp_pop = 10,
                 gt_path=r'C:\Users\zzh\MATLAB Drive\DLW-fab\gt_depth',
                 lp_path=r'C:\Users\zzh\MATLAB Drive\DLW-fab\lp_data',
                 device=torch.device('cuda'),
                 dtype=torch.float32):
        self.dev = device
        self.dtype = dtype
        self.lp_ratio=lp_ratio
        # load data
        gt = loadmat(os.path.join(gt_path,gt_file))
        lp = loadmat(os.path.join(lp_path,lp_file))
        self.x2 = lp['f0'][0][::-1]
        self.y2 = lp['mod'][0][::-1]
        # self.x2=np.append(self.x2,1.0)
        # self.y2=np.append(self.y2,0.0)
        self.y2 = self.y2*lp_ratio
        # apply compression
        self.x1 = gt['dose'][0]
        self.y1 = gt['dep_list'][0]
        # self.y1 = self.y1/max(self.y1) 
        index = self.x1<=255
        self.x1 = self.x1[index]
        self.y1 = self.y1[index]
        if self.x1[-1]<255:
            self.x1=np.append(self.x1,255)
            self.y1=np.append(self.y1,self.y1[-1])
        # initialize piecewise linear fit with your x and y data
        self.pwlf_gt = pwlf.PiecewiseLinFit(self.x1,self.y1,degree=1)
        self.pwlf_lp = pwlf.PiecewiseLinFit(self.x2,self.y2,degree=1)
        # fit the data for some line segments
        res_gt = self.pwlf_gt.fitfast(gt_seg_num, pop=gt_pop)
        res_lp = self.pwlf_lp.fitfast(lp_seg_num, pop=lp_pop)
        # save torch params
        self.params_conversion()
        # show
        if show:
            self.show()
    def params_conversion(self):
        # save lp params
        breaks = self.pwlf_lp.fit_breaks
        beta =  self.pwlf_lp.beta
        # beta
        self.beta_lp = np.ones(beta.size-1)*beta[0]
        for i in range(self.beta_lp.size):
            self.beta_lp[i] = self.beta_lp[i-1] - beta[i+1]*breaks[i]
        self.beta_lp = np.append(self.beta_lp,0)
        self.beta_lp = torch.tensor(self.beta_lp,dtype=self.dtype,device=self.dev).detach()
        # slope
        self.slope_lp = np.append(np.cumsum(beta[1:]),0)
        self.slope_lp = torch.tensor(self.slope_lp, dtype=self.dtype,device=self.dev).detach()
        # breaks
        self.breaks_lp = torch.tensor(np.append(breaks[:-1],self.x2[-1]),dtype=self.dtype,device=self.dev).detach()
        self.beta_lp.requires_grad = False
        self.slope_lp.requires_grad = False
        self.breaks_lp.requires_grad = False

        # save gt params
        breaks = self.pwlf_gt.fit_breaks
        beta =  self.pwlf_gt.beta
        # beta
        self.beta_gt = np.ones(beta.size-1)*beta[0]
        for i in range(self.beta_gt.size):
            self.beta_gt[i] = self.beta_gt[i-1] - beta[i+1]*breaks[i]
        self.beta_gt = torch.tensor(self.beta_gt,dtype=self.dtype,device=self.dev).detach()
        # slope
        self.slope_gt = torch.tensor(np.cumsum(beta[1:]),dtype=self.dtype,device=self.dev).detach()
        # breaks
        self.breaks_gt = torch.tensor(breaks[:-1],dtype=self.dtype,device=self.dev).detach()
        self.beta_gt.requires_grad = False
        self.slope_gt.requires_grad = False
        self.breaks_gt.requires_grad = False
    def show(self):
        print('LP ratio: ', self.lp_ratio)
        x1Hat = np.linspace(0, 255, num=1000)
        x2Hat = np.linspace(0, 1, num=1000)
        y1Hat_torch = self.gt_torch(torch.tensor(x1Hat,device=self.dev,dtype=self.dtype)).cpu().numpy()
        y2Hat_torch = self.lp_torch(torch.tensor(x2Hat,device=self.dev,dtype=self.dtype)).cpu().numpy()
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.x1, self.y1,'o', label='Data')
        plt.plot(x1Hat, y1Hat_torch, '-', label='Fit (torch)')
        plt.title('GT')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(self.x2, self.y2,'o', label='Data')
        plt.plot(x2Hat, y2Hat_torch, '-', label='Fit (torch)')
        plt.legend()
        plt.title('LP')
        plt.show()
    def gt(self,x):
        return np.clip(self.pwlf_gt.predict(x.ravel()),0,255).reshape(x.shape)
    def lp(self,x):
        return np.clip(self.pwlf_lp.predict(x.ravel()),0,None).reshape(x.shape)
    def lp_torch(self,x):
        s = x.ravel().unsqueeze(1)-self.breaks_lp.unsqueeze(0)
        bs = x.ravel().shape[0]
        s = torch.where(s < 0, torch.tensor(float("inf"), device=x.device), s)
        b_ids = torch.where(x.ravel()>=self.breaks_lp[0].unsqueeze(0),
                            torch.argmin(s, dim=1),
                            torch.tensor(0, device=x.device)).unsqueeze(1)
        selected_betas = torch.gather(self.beta_lp.unsqueeze(0).expand(bs, -1), dim=1, index=b_ids).squeeze(1)
        selected_slopes = torch.gather(self.slope_lp.unsqueeze(0).expand(bs, -1), dim=1, index=b_ids).squeeze(1)
        cand = selected_betas + x.ravel() * selected_slopes
        return cand.reshape(x.shape)
    def gt_torch(self,x):
        s = x.ravel().unsqueeze(1)-self.breaks_gt.unsqueeze(0)
        bs = x.ravel().shape[0]
        s = torch.where(s < 0, torch.tensor(float("inf"), device=x.device), s)
        b_ids = torch.where(x.ravel()>=self.breaks_gt[0].unsqueeze(0),
                            torch.argmin(s, dim=1),
                            torch.tensor(0, device=x.device)).unsqueeze(1)
        selected_betas = torch.gather(self.beta_gt.unsqueeze(0).expand(bs, -1), dim=1, index=b_ids).squeeze(1)
        selected_slopes = torch.gather(self.slope_gt.unsqueeze(0).expand(bs, -1), dim=1, index=b_ids).squeeze(1)
        cand = selected_betas + (x.ravel()) * selected_slopes
        return cand.reshape(x.shape)
        
def plot_and_add(pwl, func, x_start, x_end,device=torch.device('cuda')):
    '''
    plot train result for MLP fitting
    '''
    plt.figure()
    plt.xlim((x_start, x_end))
    x = torch.linspace(x_start, x_end, steps=1000).unsqueeze(1).to(device)
    y = pwl(x).cpu()
    xb, yb = get_batch(func,x_start,x_end)
    data = plt.plot(xb.squeeze(1).cpu().numpy(), yb.squeeze(1).cpu().numpy(), "xg")
    curve = plt.plot(list(x.cpu()), list(y), "b")
    plt.show()

def plot_mlp(pwl, x_start, x_end,device='cuda'):
    '''
    plot train result for MLP fitting
    '''
    plt.figure()
    plt.xlim((x_start, x_end))
    x = torch.linspace(x_start, x_end, steps=1000).unsqueeze(1).to(device)
    y = pwl(x).cpu()
    curve = plt.plot(list(x.cpu()), list(y), "b")
    plt.show()

def get_batch(func,x_start,x_end,bs=100,device='cuda'):
    '''
    generate data from fitting model to train MLP model
    '''
    x = np.random.rand(bs, 1)*(x_end-x_start) + x_start
    y = func(x.reshape(bs)).reshape(bs,1)
    return [torch.Tensor(x).to(device), torch.Tensor(y).to(device)]

def build_model(layout= [10,10,20,10,10],
                act_func=['Tanh()','Tanh()','Tanh()','Tanh()','Tanh()']):
    '''
    build MLP model
    '''
    layers = []
    layers.append(nn.Linear(1, layout[0]))
    num_of_layers = len(layout)
    for h in range(num_of_layers - 1):
        af = act_func[h]
        if af.lower() != 'none':
            layers.append(build_activation_function(af))
        layers.append(nn.Linear(layout[h], layout[h+1]))
    af = act_func[num_of_layers-1]
    if af.lower() != 'none':
        layers.append(build_activation_function(af))
    layers.append(nn.Linear(layout[num_of_layers-1], 1))
    return nn.Sequential(*layers)

def build_activation_function(af):
    exp_af = 'lambda _ : nn.' + af
    return eval(exp_af)(None)

def train_gt_mlp(num_iter = 20000, 
                  model_name = 'gt.pth',
                  model_path = '.\\config\\',
                  device = 'cuda'):
    fitting = fit_model()
    # fitting GT network
    model = build_model().to(device)
    opt = torch.optim.Adam(params=model.parameters(),
                            lr=0.001)
    loss_func = nn.MSELoss().to(device)
    model.train()
    for e in range(num_iter):
        x, y = get_batch(fitting.gt,0,255,bs=1000,device=device)
        pred = model(x)
        loss = loss_func(pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 500 ==0:
            print('Loop#',e)
            print('Loss:',loss.item())
    model.eval()
    plot_and_add(model, fitting.gt, 0, 255,device)
    torch.save(model.state_dict(), os.path.join(model_path,model_name))

def train_lp_mlp(num_iter = 10000, 
                  model_name = 'lp.pth',
                  model_path = '.\\config\\',
                  device = 'cuda'):
    fitting = fit_model()
    # fitting LP network
    model = build_model().to(device)
    opt = torch.optim.Adam(params=model.parameters(),
                            lr=0.001)
    loss_func = nn.MSELoss().to(device)
    model.train()
    for e in range(num_iter):
        x, y = get_batch(fitting.lp,0,1,bs=600,device=device)
        pred = model(x)
        loss = loss_func(pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 500 ==0:
            print('Loop#',e)
            print('Loss:',loss.item())
    model.eval()
    plot_and_add(model, fitting.lp, 0, 1,device)
    torch.save(model.state_dict(), os.path.join(model_path,model_name))

def load_mlp(model_name,
             clamp_range,
             model_path = '.\\config\\',
             device = 'cuda',
             x_min=None, x_max=None):
    model = build_model().to(device)
    model.load_state_dict(torch.load(os.path.join(model_path,model_name)))
    for param in model.parameters():
        param.requires_grad = False
    if x_min is not None and x_max is not None:
        plot_mlp(model, x_min, x_max,device)
    return lambda x: torch.clamp(model.forward(x.reshape(-1,1)),*clamp_range).reshape(x.shape)
