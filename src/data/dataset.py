import os
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from .common import numpy2tensor, load_image

class Dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to image slices.
    The user only need to rewrite the 'process_image' method.
    """

    def __init__(
        self,
        root,
        repeat=1,
        sample_rate=1.0,
        transform=None,
        save_to_memory=False
        ):
        """
        Args:
            root: Path to the dataset.
            repeat: The times to repeat the data.
            sample_rate: A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
            
        """
        self.repeat = repeat
        self.examples = [os.path.join(root, i) for i in os.listdir(root)]
        self.transform = transform
        self.save_to_memory = save_to_memory

        if self.save_to_memory:
            self.cache_image = {}

        # subsample if desired
        if sample_rate < 1.0:
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]

    def __len__(self):
        return int(self.repeat*len(self.examples))

    def __getitem__(self, i: int):
        '''
        The train and validating data is preprocessed and scaled in [0, 1]
        '''
        i = i % len(self.examples)
        if self.save_to_memory:
            if str(i) not in self.cache_image.keys():
                self.cache_image[str(i)] = self.load_single_image(i).copy()
            image = self.cache_image[str(i)]
        else:
            image = self.load_single_image(i)

        if self.transform is not None:
            transformed_batch = self.transform(image)
        else:
            transformed_batch = numpy2tensor(image)
            
        return self.examples[i], transformed_batch
    
    def load_single_image(self, i):
        image = load_image(self.examples[i])   
        return image