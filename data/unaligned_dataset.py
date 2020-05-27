import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import h5py
import numpy as np
from skimage.transform import resize as skResize
from util.util import normalize, adaptive_instance_normalization

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot_B, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_img = np.array(Image.open(A_path).convert('RGB'))
        A_img = self.stack(A_img)
        
        #Added a new loader for loading hsi images. Uncomment the following line for normal images.
        try:
            B_img = self.hsi_loader(B_path)
        except KeyError:
            print(B_path)

        B = normalize(B_img, max_=4096)
        A = normalize(A_img, max_=1)
        A = adaptive_instance_normalization(A, B)
        del A_img, B_img
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    
    def stack(self, img, resize=True):
        
        _R = img[:,:,0]
        _G = img[:,:,1]
        _B = img[:,:,2]
        
        R_img = np.stack((_R,)*10, axis=2)
        G_img = np.stack((_G,)*10, axis=2)
        B_img = np.stack((_B,)*11, axis=2)

        hsi_img = np.concatenate((B_img, G_img, R_img), axis=2)
        hsi_img = self.resize(hsi_img)
        hsi_img = np.einsum('abc->cab', hsi_img)
        return hsi_img
    
    def resize(self, img):
        img = skResize(img, (self.opt.crop_size, self.opt.crop_size))
        return img
    
    def hsi_loader(self, path):
        with h5py.File(path, 'r') as f:
            d = np.array(f['data'])
            hs_data = np.einsum('abc -> cab',self.resize(d))
        #print('Inside hsi loader, {0}'.format(np.shape(hs_data)))
        return hs_data
    
