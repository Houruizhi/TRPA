import math
import torch
import torch.autograd as autograd

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from .data.common import get_random_patch, numpy2tensor
from .data import make_dataset, numpy2tensor, tensor_split

def make_transform(dataset_cfg, train=True):

    if train:
        augment = dataset_cfg.AUGMENT
        patch_size = dataset_cfg.PATCH_SIZE
    else:
        augment = False
        patch_size = -1

    return Transform(
        augment=augment,
        if_abs=dataset_cfg.ABS,
        patch_size=patch_size,
        )

def dataloader(cfg, train=True):
    from torch.utils.data import DataLoader

    if train:
        dataset_cfg = cfg.DATASET.TRAIN

    else:
        dataset_cfg = cfg.DATASET.VAL
    
    transform = make_transform(dataset_cfg, train)

    dataset = make_dataset(dataset_cfg, transform)

    if dataset is None:
        return None
        
    if train:
        num_workers = cfg.SYSTEM.NUM_WORKERS
        loader = DataLoader(dataset=dataset, num_workers=num_workers,\
            batch_size=dataset_cfg.BATCH_SIZE, shuffle=True)
    else:
        loader = DataLoader(dataset=dataset, num_workers=1,\
            batch_size=dataset_cfg.BATCH_SIZE, shuffle=False)
    return loader

class Transform:

    def __init__(
        self,
        if_abs=False,
        patch_size=-1,
        augment=False,
        ):

        self.if_abs = if_abs
        self.patch_size = patch_size
        self.augment = augment
        self.flip = False

    def __call__(self, image):
        if self.flip and (np.random.rand(1)>0.7):
            thre = 1-0.2*np.random.rand(1)
            image_abs = np.abs(image)
            image_angle = image/(image_abs + 1e-6)
            image_abs = np.clip(image_abs, 0, thre)
            image_abs = image_abs/thre
            image = image_abs*image_angle

        if self.if_abs:
            image = np.abs(image)

        image_target = numpy2tensor(image)

        if self.patch_size > 0:
            image_target = get_random_patch(image_target, self.patch_size).permute(2,0,1)
        
        image_target = image_target / torch.sqrt(torch.sum(image_target.pow(2), dim=0, keepdim=True)).max()

        if self.augment:
            if np.random.rand(1)<0.5:
                image_target = image_target.flip(dims=[1])
            if np.random.rand(1)<0.5:
                image_target = image_target.flip(dims=[2])

        return image_target

def loss_function(scorenet, batch, max_sigma=1/3.):
    image_target = batch
    image = image_target.cuda().repeat(1,3,1,1)
    samples = image
    used_sigmas = torch.exp(torch.rand(samples.shape[0],1,1,1)*(math.log(max_sigma) - math.log(0.01)) + math.log(0.01))
    used_sigmas = used_sigmas.to(samples.device)
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    scores = scorenet(perturbed_samples, used_sigmas)

    target = - noise / (used_sigmas ** 2)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=(-3,-2,-1)) * used_sigmas.squeeze() ** 2
    with torch.no_grad():
        denoised = perturbed_samples + used_sigmas**2*scores
    return loss.sum(dim=0)/samples.shape[0], tensor_split(denoised), tensor_split(image_target)
