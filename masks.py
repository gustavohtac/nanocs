import torch
import numpy as np
from skimage.draw import line

# Define the functions with the refactored code
def radial_sampling_mask(shape, S):
    """ Radial Sampling Pattern with Sampling Rate S """
    if not (0 <= S <= 1):
        raise ValueError("Sampling rate S must be between 0 and 1")
    
    if S == 1:
        return np.ones(shape)

    total_pixels = np.prod(shape)
    sampled_pixels = int(total_pixels * S)  # Number of pixels to sample
    num_lines = sampled_pixels // shape[0]  # Approximate number of lines

    mask = np.zeros(shape)
    x_center, y_center = shape[1] // 2, shape[0] // 2  # Center coordinates

    for i in range(num_lines):
        angle = np.deg2rad(i * 360 / num_lines)
        x_end = int(x_center + np.cos(angle) * (x_center - 1))
        y_end = int(y_center + np.sin(angle) * (y_center - 1))
        rr, cc = line(y_center, x_center, y_end, x_end)
        mask[rr, cc] = 1

    return mask

def random_sampling_mask(shape, S):
    """
    Create a mask with a specified sampling rate S.

    Parameters:
    shape (tuple): Shape of the mask.
    S (float): Sampling rate, value between 0 and 1.

    Returns:
    ndarray: A mask for sampling.
    """
    if not (0 <= S <= 1):
        raise ValueError("Sampling rate S must be between 0 and 1")
    
    num_samples = int(np.prod(shape) * S)

    mask_flat = np.concatenate((np.ones(num_samples), np.zeros(np.prod(shape) - num_samples)))
    np.random.shuffle(mask_flat)
    mask = mask_flat.reshape(shape)

    return mask


def gaussian_sampling_mask(shape, S):
    """ Gaussian Sampling Pattern """
    if not (0 <= S <= 1):
        raise ValueError("Sampling rate S must be between 0 and 1")
    
    if S == 1:
        return np.ones(shape)
    
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x*x+y*y)
    # Approximately 68% of the data falls within one standard deviation (σ) of the mean
    # Approximately 95% of the data falls within two standard deviations (2σ) of the mean
    g = np.exp(-( (d)**2 / ( 2.0 * (S*2)**2 ) ) ) 
    g = (g > np.random.rand(*g.shape)).astype(int)
    return torch.from_numpy(g)


def paralel_sampling_mask(shape, acs_lines, S, pattern='random', orientation='horizontal'):
    """
    Create a mask for k-space sampling with vertical or horizontal lines based on a sampling rate S.

    Parameters:
    shape (tuple): The shape of the k-space image.
    acs_lines (int): Number of ACS (autocalibrating signal) lines for calibration.
    S (float): Sampling rate, value between 0 and 1.
    pattern (str): Pattern of sampling - 'random' or 'equispaced'.
    orientation (str): Orientation of the sampling lines - 'vertical' or 'horizontal'.

    Returns:
    Tensor: A mask for sampling the k-space image.
    """
    if not (0 <= S <= 1):
        raise ValueError("Sampling rate S must be between 0 and 1")
    
    if S == 1:
        return np.ones(shape)
    
    mask = torch.zeros(shape)

    # Handling ACS region (central region)
    if orientation == 'horizontal':
        total_lines = int(S * shape[0])
        acs_start = shape[0] // 2 - acs_lines // 2
        acs_end = acs_start + acs_lines
        mask[acs_start:acs_end, :] = 1
    else:  # vertical
        total_lines = int(S * shape[1])
        acs_start = shape[1] // 2 - acs_lines // 2
        acs_end = acs_start + acs_lines
        mask[:, acs_start:acs_end] = 1

    # Adjusting total lines by removing the ACS lines
    total_lines = max(total_lines - acs_lines, 0)

    if pattern == 'random':
        if orientation == 'horizontal':
            sample_indices = torch.randperm(shape[0])[:total_lines]
            mask[sample_indices, :] = 1
        else:  # vertical
            sample_indices = torch.randperm(shape[1])[:total_lines]
            mask[:, sample_indices] = 1
    else:  # equispaced
        if orientation == 'horizontal':
            step = max(shape[0] // total_lines, 1)
            for i in range(0, shape[0], step):
                mask[i, :] = 1
        else:  # vertical
            step = max(shape[1] // total_lines, 1)
            for i in range(0, shape[1], step):
                mask[:, i] = 1

    return mask