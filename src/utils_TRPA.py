import os
import math
import time
import numpy as np

import torch

from .data.common import tensor_split
from .metrics import batch_PSNR, batch_SSIM
from .fft import fft2, ifft2, fft2c, ifft2c

def projection_tensor(image_denoised,data_kspace,mask,rho = 2e-2):
    '''
    input shape: (n,h,w,2)
    output shape: (n,h,w,2)
    '''
    image_projection = rho*fft2c(image_denoised) + data_kspace*mask
    return ifft2c(image_projection/(rho+mask))

def denoiser(scorenet, sigma, x, step_lr=0.9, c=4):
    sigma = torch.tensor(sigma).to(x.device)
    sigma = sigma.view(1,1,1,1)
    v_var = x.repeat(1,3,1,1)
    noise = torch.randn_like(v_var).clip(-1,1) * sigma * np.sqrt(2)
    
    inputs = v_var + noise
    logp = scorenet(inputs, sigma)
    clip_c = c*sigma.squeeze().item()
    residual = step_lr*torch.clamp(logp*sigma**2,-clip_c,clip_c)
    v = x + tensor_split(residual)
    return v

def recon_TRPA(
    scorenet, 
    image_initial, 
    data_kspace, 
    mask, 
    image_target, 
    gamma=1.14,
    lam=8e-5,
    rho=0.003,
    max_iter=200,
    eps=3e-8,
    step_lr=0.43,
    c=2,
    verbose=False):
    v = image_initial
    x = v.clone()
    u = torch.zeros_like(v)

    psnrs = []
    ssims = []
    deltas = []
    rho_k = rho

    if image_target is not None:
        PSNR = batch_PSNR(image_target, v)
        SSIM = batch_SSIM(image_target, v)
        psnrs.append(PSNR)
        ssims.append(SSIM)

    time1 = time.time()
    for idx in range(max_iter):
        x_old = x.clone()
        v_old = v.clone()
        u_old = u.clone()
        #-----------------------------------------------
        # denoising step
        #-----------------------------------------------
        sigma = math.sqrt(lam/rho_k)
        with torch.no_grad():
            v = denoiser(scorenet, sigma, x + u, step_lr=step_lr, c=4)
        #-----------------------------------------------
        # projection step
        #-----------------------------------------------
        v_sub_u = v - u
        x = projection_tensor(v_sub_u.permute(0,2,3,1),data_kspace,mask,rho=1e-4*rho_k).permute(0,3,1,2)
        #-----------------------------------------------
        # multiplier update step
        #-----------------------------------------------
        u = x - v_sub_u
        if image_target is not None:
            PSNR = batch_PSNR(image_target, x)
            SSIM = batch_SSIM(image_target, x)
            psnrs.append(PSNR)
            ssims.append(SSIM)
        
        rho_k = gamma*rho_k
        
        if verbose and (idx%10 == 0):
            print(f'iter: {idx}, rho: {rho_k}, sigma: {int(sigma*255)}, PSNR: {PSNR}, SSIM: {SSIM}, TIME: {time.time()-time1}')
         
        delta = (v_old-v).pow(2).mean() + (u_old-u).pow(2).mean() + (x_old-x).pow(2).mean()
        deltas.append(delta)
        if delta < eps:
            break
    
    if image_target is not None:
        return x, (psnrs, ssims, time.time()-time1)
    else:
        return x