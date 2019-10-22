import torch
import numpy as np


def my_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different number of associated object annotations (bounding boxes).
    Arguments:
    batch: (tuple) A tuple of tensor images and lists of annotations and their ids
    Return:
    A tuple containing:
    1) (tensor) batch of images stacked on their 0 dim
    2) (list of tensors) annotations for a given image are stacked on 0 dim
    3) (list of str) id of images                                       
    """
    targets = []
    imgs = []
    for sample in batch:                                                    
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))                        
    return torch.stack(imgs, 0), targets
