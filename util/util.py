"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import h5py

def tensor2im(input_image, label=None, imtype=np.float32):
    """Converts a Tensor array into a numpy array.
    
    Parameters:
        input_image (tensor) -- the input tensor array.
        imtype (type)        -- the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy() # Convert it into a numpy array
        del image_tensor
        
        if label in ['fake_A', 'real_A', 'rec_A']:
            image_numpy = destack(image_numpy)
        if image_numpy.shape[0] == 1: # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        elif image_numpy.shape[0] == 3: # a rgb image
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        elif image_numpy.shape[0] == 31: # maybe something else, for example a hyperspectral image
            image_numpy = denormalize(image_numpy)
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def normalize(data, max_=4096, denormalize=False):
    """
    Using the ICVL BGU dataset, the max and min values were computed. 
    Normalizing to 0-1
    """
    HSI_MAX = max_
    HSI_MIN = 0

    NEW_MAX = 1
    NEW_MIN = -1
    
    scaled  = (data - HSI_MIN) * (NEW_MAX - NEW_MIN)/(HSI_MAX - HSI_MIN)  + NEW_MIN
    return scaled.astype(np.float32)

def denormalize(data, max_=4096):
    """
    Using the ICVL BGU dataset, the max and min values were computed. 
    Normalizing to 0-1
    """
    HSI_MAX = max_
    HSI_MIN = 0

    NEW_MAX = 1
    NEW_MIN = -1
    scaled = (data - NEW_MIN) * (HSI_MAX - HSI_MIN)/(NEW_MAX - NEW_MIN) + HSI_MIN 
    return scaled.astype(np.float32)

def destack(data):
    img = denormalize(data, max_=1)
    #print(np.shape(img))
    _R = np.mean(img[:11], axis=0)
    _G = np.mean(img[11:21], axis=0)
    _B = np.mean(img[21:], axis=0)
   
    hsi_img = np.array((_R, _G, _B))
    #print(np.shape(hsi_img))
    return hsi_img

def calc_mean_std(feat, eps=1e-5):
    return np.mean(feat), np.std(feat)


def adaptive_instance_normalization(content, style, channels=31):
    normalized_feat = []
    for i in range(channels):
        style_mean, style_std = calc_mean_std(style[i])
        content_mean, content_std = calc_mean_std(content[i])
        feat = (content[i] - content_mean) / content_std
        normalized_feat.append(feat * style_std + style_mean)
    normalized_feat = np.array(normalized_feat)
    #print(np.shape(normalized_feat))
    return normalized_feat