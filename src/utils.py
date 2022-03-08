import os
import cv2
import torch
import numpy as np
import random
from collections import OrderedDict
import shutil

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def get_files(root, ext = ['jpg','bmp','png']):
    files = []
    for file_ in os.listdir(root):
        file_path = os.path.join(root, file_)
        if os.path.isdir(file_path):
            files += get_files(file_path, ext)
        else:
            if file_path.split('.')[-1] in ext:
                files.append(file_path)
    return files

def clear_result_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def initial_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def gradient_clip(optimizer, gmin=-1., gmax=1.):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(gmin, gmax)
    return optimizer

def scheduler_step(scheduler, step=1):
    for _ in range(step):
        scheduler.step()
    return scheduler

def process_weights(weight_dict):
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        if k[:7] == 'module.':
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    return  new_state_dict

def load_model(model, checkpoint):
    checkpoint = process_weights(checkpoint)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    return model

def save_state_dict(model, optimizer, epoch, path):
    d = {'epoch': epoch, 'weights': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(d, path)

def load_checkpoint(path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
    else:
        print(f'{path} not exists')
        checkpoint = None
    return checkpoint