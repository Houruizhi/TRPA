import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
from common import kspace2image
import shutil
import nibabel as nib

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def split_dataset(data_dir, ext = ['.gz']):
    images = []
    for i in os.listdir(data_dir):
        if os.path.splitext(i)[1] == ".gz":
            images.append(i)
    np.random.shuffle(images)
    train_list = images[:int(0.8*len(images))]
    val_list = images[int(0.8*len(images)):int(0.9*len(images))]
    test_list = images[int(0.9*len(images)):]

    dirs = ['train', 'val', 'test']
    files_list = [train_list, val_list, test_list]

    for out_dir, file_list in zip(dirs, files_list):
        dest_dir = os.path.join(data_dir, out_dir)
        os.makedirs(dest_dir, exist_ok=True)
        for file_name in file_list:
            shutil.move(os.path.join(data_dir, file_name), os.path.join(dest_dir, file_name))


def get_files(root, ext = ['jpg','bmp','png']):
    files = []
    for file_ in os.listdir(root):
        file_path = os.path.join(root, file_)
        if os.path.isdir(file_path):
            files += get_files(file_path)
        else:
            if file_path.split('.')[-1] in ext:
                files.append(file_path)
    return files

def process_natureimg(data_path, out_path, color=True, scales=[1]):
    files = get_files(data_path)
    mkdir(out_path)
    for _, file_ in tqdm(enumerate(files)):
        file_name = file_.split('/')[-1].split('.')[0]
        data = cv2.imread(file_, 1)
        h, w, c = data.shape
        if not color:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        for scale in scales:
            if scale != 1:
                data = cv2.resize(data, (int(h*scale), int(w*scale)), interpolation=cv2.INTER_CUBIC)
                save_name = file_name+'_'+str(scale).replace('.', '')+'.npy'
            else:
                save_name = file_name+'.npy'
            np.save(os.path.join(out_path, save_name), data/255.)
    print(f'sample number: {len(files)}')

def process_fastMRI(data_path, out_path, sampling_rate=1.):
    files = get_files(data_path, 'h5')
    np.random.shuffle(files)
    files = files[:int(sampling_rate*len(files))]
    mkdir(out_path)
    for _, file_ in tqdm(enumerate(files)):
        file_name = file_.split('/')[-1].split('.')[0]
        with h5py.File(file_, 'r') as hf:
            num_slices = hf["kspace"].shape[0]
            for slice_i in range(10, num_slices):
                slice_i_str = str(slice_i)
                if len(slice_i_str) < 2:
                    slice_i_str = '0'+slice_i_str
                save_path = os.path.join(out_path, f'{file_name}_{slice_i_str}.npy')
                if os.path.exists(save_path) and (os.path.getsize(save_path) > 0):
                    continue
                image_i = kspace2image(np.array(hf["kspace"][slice_i]))
                image_i = image_i / np.abs(image_i).max()
                if np.std(image_i[20:50]) > 0.02:
                    continue
                np.save(save_path, image_i)
    print(f'sample number: {len(files)}')

def nii2npy(IMG_PATH, save_path):
    files = get_files(IMG_PATH, ['gz'])
    os.makedirs(save_path, exist_ok=True)
    num = 0
    for img_path in files:
        file_name = img_path.split('/')[-1].split('.')[0]
        img_3D = nib.load(img_path).get_fdata()
        slices_num = img_3D.shape[2]
        img_3D = img_3D/img_3D.max()
        img_3D = img_3D.astype(np.float)
        for k in range(5, slices_num):
            num += 1
            np.save(os.path.join(save_path, f'{file_name}_{k}.npy'),img_3D[:,:,k])
    
    print(f'sample number: {num}')

def mat2npy(IMG_PATH, save_path):
    import scipy.io
    files = get_files(IMG_PATH, ['mat'])
    os.makedirs(save_path, exist_ok=True)
    num = 0
    for img_path in files:
        file_name = img_path.split('/')[-1].split('.')[0]
        image = scipy.io.loadmat(img_path)['Img']
        image = image/np.abs(image).max()
        num += 1
        np.save(os.path.join(save_path, f'{file_name}.npy'),image)
    
    print(f'sample number: {num}')

def process_calgaryCampinas(data_path, out_path, sampling_rate=1.):
    files = get_files(data_path, 'npy')
    np.random.shuffle(files)
    files = files[:int(sampling_rate*len(files))]
    mkdir(out_path)
    for _, file_ in tqdm(enumerate(files)):
        file_name = file_.split('/')[-1].split('.')[0]
        kspace = np.load(file_)
        kspace_complex = kspace[...,0] + 1j*kspace[...,1]
        num_slices = kspace_complex.shape[0]
        for slice_i in range(15, num_slices):
            slice_i_str = str(slice_i)
            if len(slice_i_str) < 3:
                slice_i_str = '0'*(3-len(slice_i_str)) + slice_i_str
            save_path = os.path.join(out_path, f'{file_name}_{slice_i_str}.npy')
            if os.path.exists(save_path) and (os.path.getsize(save_path) > 0):
                continue
            image_i = np.fft.ifft2(kspace_complex[slice_i])
            image_i = image_i/np.abs(image_i).max()
            np.save(save_path, image_i)
    print(f'sample number: {len(files)}')

if __name__ == '__main__':
    # process_natureimg('/home/rzhou/DataSets/nature_img/DIV2K/DIV2K_train_HR', '/home/rzhou/ssd_cache/DIV2K_gray', False, scales=[1])
    # process_natureimg('/home/rzhou/DataSets/nature_img/TestSet/Set12', '/home/rzhou/ssd_cache/Set12', False, scales=[1])

    # process_fastMRI('/home/rzhou/local/MRIDatasets/knee_singlecoil_train/singlecoil_train', '/home/rzhou/ssd_cache/fastMRI_npy_lownoise/singlecoil_train')
    # process_fastMRI('/home/rzhou/local/MRIDatasets/knee_singlecoil_train/singlecoil_val', '/home/rzhou/ssd_cache/fastMRI_npy_lownoise/singlecoil_val', 0.06)
    
    # nii2npy('/home/rzhou/DataSets/MRI/IXI-T1/train', '/home/rzhou/ssd_cache/IXI-T1/train')
    # nii2npy('/home/rzhou/DataSets/MRI/IXI-T1/val', '/home/rzhou/ssd_cache/IXI-T1/val')
    # nii2npy('/home/rzhou/DataSets/MRI/IXI-T1/test', '/home/rzhou/ssd_cache/IXI-T1/test')
    
    # mat2npy('/home/rzhou/local/SIAT/train', '/home/rzhou/ssd_cache/SIAT/train')
    # mat2npy('/home/rzhou/local/SIAT/test', '/home/rzhou/ssd_cache/SIAT/test')
   
    # process_calgaryCampinas('/home/rzhou/local/calgary-campinas-singlecoil/Train', '/home/rzhou/ssd_cache/calgary-campinas-singlecoil/tain', sampling_rate=1.)
    # process_calgaryCampinas('/home/rzhou/local/calgary-campinas-singlecoil/Val', '/home/rzhou/ssd_cache/calgary-campinas-singlecoil/val', sampling_rate=1.)