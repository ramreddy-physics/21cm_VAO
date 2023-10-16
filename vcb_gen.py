import numpy as np
import py21cmfast as p21c
import os
import h5py
from shutil import rmtree

HII_DIM = 16
BOX_LEN = 25

psi=1.3             #variable that defines the wedge

def remove_wedge(image: np.array):
    
    #For N_x = N_y = N_z
    
    f1=np.fft.fftn(image)

    def rule_remove_wedge(i, j, k):
        return 0 if i!=0 and j!=0 and k!=0 and i/np.sqrt(j**2 + k**2) % 2 < np.tan(psi) else f1[i, j, k]

    rule_vectorized = np.vectorize(rule_remove_wedge)
    shape = image.shape
    indices = [np.arange(shape[0])[:, np.newaxis, np.newaxis], np.arange(shape[1])[:, np.newaxis], np.arange(shape[2])]
    
    f2 = rule_vectorized(*indices)

    return np.real(np.fft.ifftn(f2))


user_params = {
    "HII_DIM": HII_DIM,
    "BOX_LEN": BOX_LEN,
    "USE_FFTW_WISDOM": True,
    "USE_INTERPOLATION_TABLES": True,
    "FAST_FCOLL_TABLES": True,
    "USE_RELATIVE_VELOCITIES": True,
    "POWER_SPECTRUM": 5,
}


def create_init_box(random_seed):
    return p21c.initial_conditions(
        user_params=user_params,
        random_seed=random_seed,
        direc='_cache'
    )


def generate_vcb(num_cubes: int, include_wr: bool = True):
    
    images = np.array([create_init_box(random_seed=i+1).lowres_vcb for i in range(num_cubes)])

    if include_wr == True:
        return images, np.array([remove_wedge(image) for image in images])
    
    else:
        return images, images
    