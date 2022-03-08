import numpy as np
import torch

from .common import (
    load_mask,
    augment_img,
    random_augment_img, 
    get_random_patch, 
    numpy2tensor, 
    generate_gaussian_noise,
    kspace2image,
    image2kspace
) 


def make_transform(dataset_cfg, train=True):
    
    if dataset_cfg.TASK == 'denoise':
        transform = make_denoising_transform(dataset_cfg, train)
    elif dataset_cfg.TASK == 'reconstruction': 
        transform = make_recon_transform(dataset_cfg, train)
    elif dataset_cfg.TASK == 'null': 
        transform = None
    else:
        raise NotImplementedError('There are not Dataset implemented for %s'%(dataset_cfg.TASK))
    
    return transform


def make_recon_transform(dataset_cfg, train=True):

    if train:
        augment = dataset_cfg.AUGMENT
    else:
        augment = False
   
    mask = load_mask(dataset_cfg.MASK_PATH)
    
    return ReconTransform(
        mask=mask,
        if_abs=dataset_cfg.ABS,
        patch_size=dataset_cfg.PATCH_SIZE,
        augment=augment,
        noisel_level=dataset_cfg.NOISE_LEVEL
        )


def make_denoising_transform(dataset_cfg, train=True):

    noise_level = dataset_cfg.NOISE_LEVEL
    patch_size = dataset_cfg.PATCH_SIZE
    noise2noise = dataset_cfg.NOISE2NOISE

    if train:
        augment = dataset_cfg.AUGMENT
        if_blind = dataset_cfg.BLIND
        seed = None
    else:
        augment = False
        if_blind = False
        seed = dataset_cfg.SEED

    return DenoiseTransform(
        noise_level=noise_level,
        if_blind=if_blind,
        noise2noise=noise2noise,
        augment=augment,
        if_abs=dataset_cfg.ABS,
        patch_size=patch_size,
        seed=seed
        )


class ReconTransform:

    def __init__(
        self,
        mask=None,
        if_abs=False,
        patch_size=-1,
        augment=False,
        noisel_level=0
        ):
        '''
        low frequency is centered
        '''
        self.if_abs = if_abs
        self.augment = augment
        self.mask = mask
        self.patch_size = patch_size
        self.noisel_level = noisel_level

    def __call__(self, image_target): 
        
        if not np.iscomplexobj(image_target):
            image_target = image_target.astype(np.complex)

        if self.patch_size >= 0:
            image_target = get_random_patch(image_target, self.patch_size)
        
        assert image_target.shape[-2:] == self.mask.shape[-2:]
        
        if self.augment:
            mode = np.random.randint(0, 8)
            image_target = augment_img(image_target, mode)

        kspace = image2kspace(image_target)
        if self.noisel_level > 0:
            noise_level = self.noisel_level*np.max(np.abs(kspace))
            # noise_level = 10.
            kspace = kspace + noise_level*(np.random.randn(*kspace.shape) + 1j*np.random.randn(*kspace.shape))
        
        kspace_sampled = self.mask*kspace
        image_zerofilled = kspace2image(kspace_sampled)

        if self.if_abs:
            image_target = np.abs(image_target)
            image_zerofilled = np.abs(image_zerofilled)

        image_target = numpy2tensor(image_target)
        kspace_sampled = numpy2tensor(kspace_sampled)
        image_noisy = numpy2tensor(image_zerofilled)
        mask = numpy2tensor(self.mask)
        kspace_sampled = mask*kspace_sampled
        return image_target, image_noisy, kspace_sampled, mask


class DenoiseTransform:

    def __init__(
        self,
        noise_level,
        if_blind=False,
        if_abs=False,
        patch_size=-1,
        noise2noise=False,
        augment=False,
        seed=None
        ):

        self.if_abs = if_abs
        self.noise_level = noise_level/255.
        self.if_blind = if_blind
        self.patch_size = patch_size
        self.augment = augment
        self.noise2noise = noise2noise
        self.seed = seed
        
    def __call__(self, image):
        if self.seed:
            torch.manual_seed(self.seed)

        std = np.std(image[:100,:])
        if self.patch_size > 0:
            image = get_random_patch(image, self.patch_size)

        if self.augment:
            image = random_augment_img(image)

        if self.if_abs:
            image = np.abs(image)

        image_target = numpy2tensor(image)

        if self.if_blind:
            noise_level = self.noise_level*np.random.rand(1).clip(1e-2, 1).item()
        else:
            noise_level = self.noise_level
            
        image_target = image_target + generate_gaussian_noise(image_target.size(), 0, std)

        image_noisy = image_target + generate_gaussian_noise(image_target.size(), 0, noise_level)

        if self.noise2noise: 
            image_noisy = image_noisy + generate_gaussian_noise(image_noisy.size(), 0, noise_level)
            image_target = image_target + generate_gaussian_noise(image_target.size(), 0, noise_level)

        return image_target, image_noisy, torch.tensor(noise_level).view(1,1,1)


