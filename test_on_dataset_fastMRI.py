#!/usr/bin/env python
# coding: utf-8

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from src.utils_TRPA import recon_TRPA
from src.fastmri.subsample import RandomMaskFunc
from src.fft import fft2c, ifft2c
from src.data import numpy2tensor, tensor2image, tensor2complex, tensor_split, sum_multichannel, kspace2image
from src.metrics import psnr, ssim, batch_PSNR

from scipy.io import loadmat,savemat

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from src.models.condrefinenet import CondRefineNetDilated
chp_path = './checkpoints/fastMRI/net.pth'
states = torch.load(chp_path)
ch=6
scorenet = CondRefineNetDilated(ch,ch,128).cuda()
scorenet.load_state_dict(states['weights'])
scorenet.eval()

def crop_image(image, win = 320):
    n,c,h,w = image.shape
    return image[...,h//2-win//2:h//2+win//2,w//2-win//2:w//2+win//2]

mask_params = [(4,0.08), (8,0.04)]
for mask_parm in mask_params:
    mask_func = RandomMaskFunc(
        center_fractions=[mask_parm[1]],
        accelerations=[mask_parm[0]]
    )
    factor = f'x{mask_parm[0]}'
    data_dir = f'/home/rzhou/ssd_cache/fastMRI_npy/singlecoil_val/'
    save_dir = './results/fastMRI/TRPA'
    files = os.listdir(data_dir)
    res_save_dir = os.path.join(save_dir, factor)
    zero_filled_path = save_dir.replace('TRPA', f'Zero-filled/{factor}')
    os.makedirs(zero_filled_path, exist_ok=True)
    os.makedirs(res_save_dir, exist_ok=True)

    psnr_avg = 0
    for kk, file_ in enumerate(files):
        if os.path.exists(os.path.join(res_save_dir,file_.split('.')[0]+'.mat')):
            continue
        
        img_original = np.load(os.path.join(data_dir, file_))
        n2nstd = np.std(img_original[:80,:])
            
        if n2nstd > 0.02:
            gamma=1.09
            lam=4e-4
            rho=0.003
            max_iter=100
            eps=1e-4
            step_lr=0.5
            c=3
        else:
            gamma=1.07
            lam=4e-4
            rho=0.0035
            max_iter=100
            eps=1e-5
            step_lr=0.6
            c=3
       
    
        kspace = fft2c(numpy2tensor(img_original))
        mask = mask_func(kspace.shape, 1234)
        kspace_sampled = np.multiply(mask, kspace)
        image_zeroFilled = ifft2c(kspace_sampled)
        image_zeroFilled = image_zeroFilled.unsqueeze(0).permute(0,3,1,2)
        # file name: test_data_26.npy, PSNR: 33.64713609203101
        image_target = numpy2tensor(img_original).permute(2,0,1).unsqueeze(0) # only compute the PSNR

        image_initial = image_zeroFilled

        data_kspace = kspace_sampled.unsqueeze(0) # (1,h,w,2)

        mask = torch.Tensor(mask).unsqueeze(0) #(1,h,w,1)

        image_initial, data_kspace, mask = image_initial.cuda(), data_kspace.cuda(), mask.cuda()
#             rec_im = image_initial
        rec_im, _ = recon_TRPA(
            scorenet, 
            image_initial, 
            data_kspace, 
            mask, 
            image_target, 
            gamma=gamma,
            lam=lam,
            rho=rho,
            max_iter=max_iter,
            eps=eps,
            step_lr=step_lr,
            c=c,
            verbose=False)
        savemat(os.path.join(res_save_dir,file_.split('.')[0]+'.mat'),{'rec_im':tensor2complex(rec_im)[0]})
        PSNR = batch_PSNR(crop_image(rec_im), crop_image(image_target))
        psnr_avg += PSNR
        print(kk, f'{file_}, PSNR: {PSNR}')
        
        savemat(os.path.join(zero_filled_path, file_.split('.')[0]+'.mat'), {'rec_im': tensor2image(image_zeroFilled)[0]})
    print(mask_parm, 'average psnr: ', psnr_avg/len(files))