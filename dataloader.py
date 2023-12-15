import os
import h5py
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from fastprogress import progress_bar
from masks import *
import random
# import sigpy as sp

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


class MRIDataset(Dataset):
    """
    A custom dataset class for loading and processing MRI scan data from HDF5 files.

    This class is designed to handle MRI data stored in HDF5 format, where each file
    contains multiple slices of k-space data and root-sum-square (RSS) reconstructions.
    It supports indexing to retrieve individual slices across all scans in a specified
    directory.

    Attributes:
        file_paths (list): A list of paths to HDF5 files in the given folder.
        num_slices (numpy.array): An array where each element represents the number of slices 
                                  in the corresponding scan file.
        slice_mapper (numpy.array): A cumulative index array for mapping global slice indices
                                    to specific scans and local slice indices.

    Methods:
        __len__(): Returns the total number of slices in the dataset.
        __getitem__(idx): Retrieves the RSS reconstruction of the slice corresponding to the 
                          global index `idx`.
        read_hdf5_file(file_path, slice_idx): Static method to read k-space and RSS data 
                                              from a HDF5 file for a given slice index.

    Parameters:
        folder_path (str): Path to the directory containing HDF5 files of MRI scans.

    Example:
        >> dataset = MRIDataset('/path/to/mri/scans')
        >> first_scan_rss = dataset[0]['reconstruction_rss']
    
    Note:
        - This class assumes that each HDF5 file has 'kspace' and 'reconstruction_rss' datasets.
        - Works with fastMRI dataset https://fastmri.med.nyu.edu/

    Based on:
        https://github.com/utcsilab/csgm-mri-langevin/blob/main/dataloaders.py
    """
    def __init__(self, folder_path, image_size=(320,320)):

        self.image_size = image_size
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
        
        # Access meta-data of each scan to get number of slices
        self.num_slices = np.zeros((len(self.file_paths,)), dtype=int)
        for idx, file in progress_bar(enumerate(self.file_paths), total=len(self.file_paths)):
            with h5py.File(os.path.join(file), 'r') as data:
                self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    def __getitem__(self, idx):
        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)
        
        file_path = self.file_paths[scan_idx]

        # Read desired image
        _, k_spaces, reconstruction_rss = self.read_hdf5_file(file_path, slice_idx)

        # Apply image transformations (reconstruction_rss)
        transform_nn = transforms.Compose([
            transforms.ToTensor(),                 # to range [0.0, 1.0]
            transforms.Resize(self.image_size, antialias=True),
            transforms.Lambda(lambda x: ((x - x.min()) / (x.max() - x.min()))),
            transforms.Normalize((0.5,), (0.5,)),  # range [-1, 1]
        ])
        reconstruction_rss = transform_nn(reconstruction_rss)

        # Apply k_spaces transformations
        # reconstructed_kspace = self.rss_reconstruction(k_spaces)

        degraded_measurement = self.degrade_with_mask(reconstruction_rss)

        degraded_measurement = transform_nn(degraded_measurement)
        
        return {
            'images': reconstruction_rss.float(), 
            'images_degraded': degraded_measurement.float(), 
            # 'k_spaces': k_spaces
        }
    
       
    @staticmethod
    def read_hdf5_file(file_path, slice_idx):
        """
        Reads an HDF5 file containing MRI data and extracts specific information.

        This method opens an HDF5 file specified by the file_path and reads data related to MRI. 
        It extracts the ISMRMRD header, a specific slice of k-space data, and a corresponding 
        slice of the reconstruction RSS (Root-Sum-of-Squares) image. The ISMRMRD header is 
        converted from bytes to a string and then parsed into an XML element tree. The k-space 
        and reconstruction RSS data for the specified slice index are also extracted and converted 
        to NumPy arrays.

        Parameters:
        file_path (str): The path to the HDF5 file to be read.
        slice_idx (int): The index of the slice to extract from the k-space and reconstruction RSS datasets.

        Returns:
        tuple: A tuple containing three elements in the following order:
            - An XML element tree representing the ISMRMRD header.
            - A NumPy array containing the k-space data for the specified slice.
            - A NumPy array containing the reconstruction RSS data for the specified slice.
        """
        with h5py.File(file_path, 'r') as file:
            ismrmrd_header = file['ismrmrd_header'][()]
            if isinstance(ismrmrd_header, bytes):
                ismrmrd_header = ismrmrd_header.decode('utf-8')   
            header_xml = ET.fromstring(ismrmrd_header)

            # Read kspace data
            k_spaces = np.array(file['kspace'])[slice_idx]
            
            # Read and Resize reconstruction_rss
            reconstruction_rss = np.array(file['reconstruction_rss'])[slice_idx]
            
        return header_xml, k_spaces, reconstruction_rss
    

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        """
        Crops an input array (image or tensor) to the specified dimensions centered on both axes.

        The cropping is centered, meaning that equal amounts are cropped from opposite sides of each 
        dimension if possible. 

        Parameters:
        x (array-like): The input array or tensor to be cropped. The spatial dimensions to be cropped 
                        are expected to be the last two dimensions of 'x'.
        wout (int): The desired width of the output cropped array.
        hout (int): The desired height of the output cropped array.

        Returns:
        array-like: The cropped portion of the input array. The output will have the same number of 
                    dimensions as the input and will be of size 'wout' x 'hout' in its last two dimensions.
        """
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]
   
    def rss_reconstruction(self, k_spaces):
        """
        Reconstruct an image using Root-Sum-of-Squares (RSS) from multiple k-space datasets.
        
        :param k_spaces: A list of k-space datasets (each is a 2D numpy array)
        :return: Reconstructed image
        """
        image_sum_of_squares = np.zeros_like(k_spaces[0], dtype=np.float64)

        for k_space in k_spaces:
            # Apply Inverse Fourier Transform to each k-space
            image = np.fft.ifft2(k_space)
            image_abs = np.abs(image)

            # Sum of squares
            image_sum_of_squares += image_abs ** 2

        # Root of the sum of squares
        rss_image = np.sqrt(image_sum_of_squares)
        
        # Shift the image to be centered
        rss_image = np.fft.fftshift(rss_image)
        
        return rss_image
    
    def _get_mask(self, R=0.5, pattern='random'):

        if pattern == 'random':
            mask = random_sampling_mask(self.image_size, R)
        elif pattern == 'radial':
            mask = radial_sampling_mask(self.image_size, R)
        elif pattern == 'gaussian':
            mask = gaussian_sampling_mask(self.image_size, R)
        elif pattern == 'horizontal':
            mask = paralel_sampling_mask(self.image_size, 30, R, pattern='random', orientation='horizontal')
        elif pattern == 'vertical':
            mask = paralel_sampling_mask(self.image_size, 30, R, pattern='random', orientation='vertical')
        else:
            raise NotImplementedError('Mask Pattern not implemented yet...')

        return mask
    
    def degrade_with_mask(self, image):
        # Randomly select a value for R and a pattern
        R = random.uniform(0.1, 0.9)
        patterns = ['random', 'radial', 'gaussian', 'horizontal', 'vertical']
        pattern = random.choice(patterns)

        # Degrade
        k_space = fft2(image)
        mask = self._get_mask(R=R, pattern=pattern)
        masked_k_space = k_space * mask
        degraded_image = torch.real(ifft2(masked_k_space))
        
        return degraded_image.squeeze().numpy()
    

    # def apply_mask_and_reconstruct(self, kspace_data, total_lines, R, pattern, orientation):
    #     """
    #     Applies a k-space sampling mask to the provided k-space data and reconstructs the image 
    #     using root-sum-of-squares (RSS) reconstruction.

    #     Parameters:
    #     kspace_data (numpy array): The k-space data with dimensions (n_coils, w, h).
    #     acs_lines (int): The number of Autocalibration Signal (ACS) lines for mask generation.
    #     total_lines (int): The total number of lines in k-space along the phase-encoding direction.
    #     R (float): The acceleration factor for mask generation.
    #     pattern (str): The pattern of the mask ('random' or 'equispaced').
    #     rss_reconstruction (function): The function to perform RSS reconstruction.

    #     Returns:
    #     numpy array: The reconstructed image using RSS method.
    #     """     
    #     # Reduce FoV by half in the readout direction
    #     kspace_data = sp.ifft(kspace_data, axes=(-2,))
    #     kspace_data = sp.resize(kspace_data, (kspace_data.shape[0], 320,
    #                                 kspace_data.shape[2]))
    #     kspace_data = sp.fft(kspace_data, axes=(-2,)) # Back to k-space

    #     # Compute ACS size based on R factor and sample size
    #     total_lines = kspace_data.shape[-1]

    #     if 1 < R <= 6:
    #         # Keep 8% of center samples
    #         acs_lines = np.floor(0.08 * total_lines).astype(int)
    #     else:
    #         # Keep 4% of center samples
    #         acs_lines = np.floor(0.04 * total_lines).astype(int)
        
    #     # Generate the mask
    #     mask = self._get_mask(acs_lines, total_lines, R, pattern)

    #     # Mask k-space
    #     if orientation == 'vertical':
    #         kspace_data *= mask[None, None, :]
    #     elif orientation == 'horizontal':
    #         kspace_data *= mask[None, :, None]
    #     else:
    #         raise NotImplementedError

    #     # Perform RSS reconstruction
    #     reconstructed_image = self.rss_reconstruction(kspace_data)

    #     return kspace_data, reconstructed_image