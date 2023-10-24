from vcb_gen import generate_vcb
from modules import UNet
import numpy as np

import torch
import torch.nn as nn

from py21cmfast import global_params
from py21cmfast import plotting


N_test = 25
channel_list = [1, 4, 4, 4, 8, 16]

vcb, vcb_wr = generate_vcb(num_cubes=N_test)

X = torch.tensor(np.reshape(vcb_wr, (N_test, 1, 16, 16, 16)), dtype=torch.float32)
Y = torch.tensor(np.reshape(vcb, (N_test, 1, 16, 16, 16)), dtype=torch.float32)


Model = UNet(channel_list=channel_list)
Model = torch.load("UNet_vcb_2.pt", map_location=torch.device("cpu"))

vcb_pred = np.reshape(Model(X).detach().numpy(), (N_test, 16, 16, 16))

from sklearn.metrics import mean_squared_error as mse

print(
    "mean squared error on testing dataset = %f"
    % mse(vcb.reshape(N_test, 16**3), vcb_pred.reshape(N_test, 16**3))
)  # between original and reconstructed
