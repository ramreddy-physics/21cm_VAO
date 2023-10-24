import numpy as np
import py21cmfast as p21c

from py21cmfast import global_params
from py21cmfast import plotting
from kernel import Kernel

import torch
import torch.nn as nn

from vcb_gen import generate_vcb
from modules import UNet

N = 100

vcb, vcb_wr = generate_vcb(num_cubes=N)
channel_list = [1, 4, 4, 8, 8, 16]
Model = UNet(channel_list=channel_list)

X = torch.tensor(np.reshape(vcb_wr, (N, 1, 16, 16, 16)), dtype=torch.float32)
Y = torch.tensor(np.reshape(vcb, (N, 1, 16, 16, 16)), dtype=torch.float32)

Train_kernel = Kernel(Model=Model, X=X, Y=Y)

Train_kernel.train_model(EPOCHS=1500)
