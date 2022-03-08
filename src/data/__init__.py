from .common import (
    load_mask,
    load_image, 
    complex2tensor, 
    image2tensor, 
    tensor2image, 
    sum_multichannel,
    numpy2tensor, 
    tensor2complex, 
    kspace2image,
    image2kspace,
    generate_cartesian_mask,
    tensor_split)
from .transforms import make_transform
from .dataset import Dataset

def make_dataset(dataset_cfg, transform):

    if dataset_cfg.ROOT == '':
        return None

    return Dataset(
        root=dataset_cfg.ROOT,
        repeat=dataset_cfg.REPEAT,
        sample_rate=dataset_cfg.SAMPLE_RATE,
        transform=transform,
        save_to_memory=dataset_cfg.SAVE_TO_MEMORY
    )


def make_train_dataloader(dataset_cfg, train=True):
    from torch.utils.data import DataLoader
    
    transform = make_transform(dataset_cfg, train)
    dataset = make_dataset(dataset_cfg, transform)

    if dataset is None:
        return None
        
    if train:
        loader = DataLoader(dataset=dataset, num_workers=4,\
            batch_size=dataset_cfg.BATCH_SIZE, shuffle=True)
    else:
        loader = DataLoader(dataset=dataset, num_workers=1,\
            batch_size=dataset_cfg.BATCH_SIZE, shuffle=False)
    return loader