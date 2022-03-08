import os
import cv2
import torch
import numpy as np
from scipy.io import loadmat
from numpy.random import randint

def load_image(file_path, color=False):
    _, file_ext = os.path.splitext(file_path)
    if file_ext == '.npy':
        data = np.load(file_path)
        if data.dtype in [np.complex64, np.complex128]:
            return data
        else:
            return data
    elif file_ext in ['.tif','.jpg','.png','.bmp']:
        data = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if not color:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        return data/255.
    else:
        raise NotImplementedError('There is no loading method implemented for %s files' %(file_ext))

def load_mask(file_path):
    _, file_ext = os.path.splitext(file_path)
    if file_ext == '.npy':
        mask = np.load(file_path)
    elif file_ext in ['.tif','.jpg','.png','.bmp']:
        mask = cv2.imread(file_path, 0)
    elif file_ext == '.mat':
        mask = loadmat(file_path)['mask']
    else:
        raise NotImplementedError('There is no loading method implemented for %s files' %(file_ext))
    
    return mask/mask.max()
    
def random_augment_img(img):
    mode = randint(0, 8)
    return augment_img(img, mode)
    
def augment_img(img, mode):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def crop_image(img, crop_h=(0,0), crop_w=(0,0)): 
    '''
    input img size: (w,h,3) or (w,h)
    Args:
        crop_h: the begin and end location along the height axis.
        crop_w: the begin and end location along the width axis.
    '''
    assert 2 <= len(img.shape) <= 3
    return img[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1], ...]

def center_crop_image(img, win_size=(0,0)): 
    '''
    input img size: (w,h,3) or (w,h)
    Args:
        win_size: the size of the cropped window.
    '''
    h_c = img.shape[0]//2
    w_c = img.shape[1]//2
    win_h = win_size[0]//2
    win_w = win_size[1]//2
    return crop_image(img, (h_c-win_h,h_c+win_h), (w_c-win_w,w_c+win_w))

def get_random_patch(img, patch_size):
    if patch_size < 0:
        return img
    assert 2 <= len(img.shape) <= 3
    # np.random.randint(a,b) returns random integer in range [a, b)
    hi = randint(0, img.shape[0] - patch_size + 1) 
    wi = randint(0, img.shape[1] - patch_size + 1)
    img = crop_image(img, (hi, hi + patch_size), (wi, wi + patch_size))
    return img

def generate_gaussian_noise(size, mean, std):
    return torch.FloatTensor(size).normal_(mean=mean, std=std)

def complex2tensor(image, dtype=torch.float32):
    image_real = torch.tensor(image.real, dtype=dtype).unsqueeze(-1)
    image_imag = torch.tensor(image.imag, dtype=dtype).unsqueeze(-1)
    return torch.cat((image_real, image_imag), dim=-1)

def image2tensor(image, dtype=torch.float32):
    image = torch.tensor(image, dtype=dtype)
    assert 2 <= len(image.shape) <=3
    if len(image.shape) == 2:
        image = image.unsqueeze(-1)
    return image

def numpy2tensor(image, dtype=torch.float32):
    
    if np.iscomplexobj(image):
        image = complex2tensor(image.copy(), dtype=dtype)
    else:
        image = image2tensor(image.copy(), dtype=dtype)

    return image

def sum_multichannel(img):
    """
    input: (n,c,w,h)
    output: (n,1,w,h)
    """ 
    c = img.shape[1]
    assert c%2 == 0
    return torch.sqrt(torch.sum(img**2, dim=1, keepdim=True)/(c//2))

def tensor2image(img):
    """
    input: (n,c,w,h)
    output: (n,w,h)
    """ 
    c = img.shape[1]

    assert (c <= 3) or (c%2 == 0)
    
    if c % 2 == 0:
        img = sum_multichannel(img).squeeze(1)
    elif c == 1:
        img = img.squeeze(1)
    elif c == 3:
        img = img.permute(0,2,3,1)
    if img.__class__ == torch.Tensor:
        if img.device != torch.device('cpu'):
            img = img.cpu()
        return img.detach().numpy().astype(np.float32)

def tensor2complex(img):
    """
    input: (n,2,w,h)
    output: (n,w,h)
    """ 
    c = img.shape[1]

    assert c == 2

    if img.__class__ == torch.Tensor:
        if img.device != torch.device('cpu'):
            img = img.cpu()
    img = img.numpy().astype(np.float32)
    return img[:,0,...] + 1j*img[:,1,...]

def kspace2image(kspace, norm=None):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(kspace), norm=norm))

def image2kspace(image, norm=None):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image), norm=norm))

def generate_cartesian_mask(shape, center_fraction, ratio):
    '''
    apply the mask to k-space data that the low frequency is located in corner. 
    '''
    assert len(shape) == 2
    num_cols = shape[-1]
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols * ratio - num_low_freqs) / (num_cols - num_low_freqs)
    index = np.random.rand(num_cols) < prob
    pad_loc = (num_cols - num_low_freqs + 1) // 2
    index[pad_loc:pad_loc+num_low_freqs] = True
    mask = np.zeros(shape)
    mask[:,index] = 1
    return mask


def tensor_split(tensor, dim=1, each_channel=2):
    tensor_split = torch.split(tensor.unsqueeze(-1), each_channel, dim=dim)
    return torch.cat(tensor_split, dim=-1).mean(-1)
    
if __name__ == '__main__':
    mask = generate_cartesian_mask((640, 368), 0.08, 0.6)
    print(mask.sum()/368)