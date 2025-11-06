import numpy as np
#employed in https://www.sciencedirect.com/science/article/pii/S0889157523001199?via%3Dihub

def normalize_by_spectral(Cube, integral_type='L1'):
    """
    Normalize each pixel’s spectrum by its integrated spectrum (L1 or L2 norm),
    then apply a scale factor.

    Parameters:
        Cube (numpy.ndarray): Hyperspectral cube of shape (H, W, B)
        integral_type (str): 'L1' or 'L2'

    Returns:
        Cube_out (numpy.ndarray): Normalized and scaled hyperspectral cube
    """
    eps_val = 1e-12
    scale_factor = 100.0

    Cube = np.maximum(Cube, 0)  # ensure non-negative values

    if integral_type == 'L1':
        SpecInt = np.sum(Cube, axis=2)          # L1 norm per pixel
    elif integral_type == 'L2':
        SpecInt = np.sqrt(np.sum(Cube ** 2, axis=2))  # L2 norm per pixel
    else:
        raise ValueError("integral_type must be 'L1' or 'L2'")

    Den = np.maximum(SpecInt, eps_val)
    Den = np.repeat(Den[:, :, np.newaxis], Cube.shape[2], axis=2)

    Cube_out = (Cube / Den) * scale_factor

    return Cube_out
