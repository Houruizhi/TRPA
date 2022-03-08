#!/usr/bin/env python
# coding: utf-8

import os
import time
import torch
import torch.nn as nn
import numpy as np
import math

from src.models.condrefinenet import CondRefineNetDilated
from src.data import numpy2tensor, tensor2image, load_mask
from src.utils_TRPA import recon_TRPA
from src.fft import fft2c, ifft2c
from scipy.io import loadmat,savemat


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

chp_path = './checkpoints/SIAT/net.pth'
states = torch.load(chp_path)
scorenet = CondRefineNetDilated(6,6,128).cuda()
scorenet.load_state_dict(states['weights'])
scorenet.eval()

def generate_test_data(img_original, mask):
    kspace = fft2c(img_original)
    kspace_sampled = mask * kspace
    image_zeroFilled = ifft2c(kspace_sampled)
    return kspace_sampled, image_zeroFilled

noisy = True
psnr_avg = 0
data_root = './data/SIAT/test'
mask_root = './data/masks'
save_root = './results//SIAT/radial_10'
files = os.listdir(data_root)

def select_param(ratio=0.1):
    gamma=1.15
    lam=1e-4
    rho=0.003
    max_iter=100
    eps=6e-9
    step_lr=1
    c=3
    return gamma, lam, rho, max_iter, eps, step_lr, c

for mask_type in os.listdir(mask_root):
    gamma, lam, rho, max_iter, eps, step_lr, c = select_param(ratio=float(mask_type.split('.')[0][-3:])/100.)
    mask_np = load_mask(os.path.join(mask_root, mask_type))
    mask = numpy2tensor(mask_np).unsqueeze(0).cuda()

    save_path = os.path.join(save_root, mask_type.split('.')[0])
    os.makedirs(save_path, exist_ok=True)
    avg_psnr = 0
    for file_ in files:
        save_image_path = os.path.join(save_path, file_.replace('.npy', '.mat'))
        img_original = np.load(os.path.join(data_root, file_))
        img_original = img_original/np.max(np.abs(img_original))
        img_original = numpy2tensor(img_original).unsqueeze(0).cuda()
        kspace_sampled, image_zeroFilled = generate_test_data(img_original, mask)
        
        image_target, image_initial = [i.permute(0,3,1,2) for i in [img_original, image_zeroFilled]]
        
        rec_im, (psnrs, ssims, TIME) = recon_TRPA(
            scorenet, 
            image_initial, 
            kspace_sampled, 
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
    
        rec_im = tensor2image(rec_im)[0]
        savemat(save_image_path, {'rec_im': rec_im})
        
        avg_psnr += psnrs[-1]
        print(f'mask: {mask_type}, file name: {file_}, psnr: {psnrs[-1]}')
    print(f'mask: {mask_type}, average PSNR: {avg_psnr/len(files)}')