import numpy as np
from .data import tensor2image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def mse(image_target, image):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((image_target - image) ** 2)

def nmse(image_target, image):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(image_target - image) ** 2 / np.linalg.norm(image_target) ** 2

def psnr(image_target, image, data_range=1.):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(image_target, image, data_range=data_range)

def ssim(image_target, image, data_range=1.):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        image_target, image, multichannel=False, data_range=data_range
    )

def batch_PSNR(Iclean, Img, data_range=1.):
    Img = tensor2image(Img).clip(0., 1.)
    Iclean = tensor2image(Iclean).clip(0., 1.)
    res = 0
    for i in range(Img.shape[0]):
        res += psnr(Iclean[i, ...], Img[i, ...], data_range)
    return res/Img.shape[0]

def batch_SSIM(Iclean, Img, data_range=1.):
    Img = tensor2image(Img).clip(0., 1.)
    Iclean = tensor2image(Iclean).clip(0., 1.)
    res = 0
    for i in range(Img.shape[0]):
        res += ssim(Iclean[i,...], Img[i,...], data_range)
    return res/Img.shape[0]