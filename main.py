import numpy as np
import py21cmfast as p21c

from py21cmfast import global_params
from py21cmfast import plotting

import torch
import torch.nn as nn

from vcb_gen import generate_vcb
from modules import BasicUNet


#####################################################################################
N_train=25
N_test=5
N = N_train+N_test
#generate and store cubes
vcb, vcb_wr = generate_vcb(num_cubes=N)

X_train=torch.tensor(np.reshape(vcb_wr[:N_train], (N_train, 1, 16, 16, 16)), dtype=torch.float32)
Y_train=torch.tensor(np.reshape(vcb[:N_train], (N_train, 1, 16, 16, 16)), dtype=torch.float32)


####################### simple linear transform to [-1, 1] (not implemented) #################
X_max=torch.max(X)
X_min=torch.min(X)

Y_max=torch.max(Y)
Y_min=torch.min(Y)

def normalize(T, T_max, T_min):
    a=2.0/(T_max - T_min)
    b=-(T_max + T_min)/(T_max - T_min)
    return a*T + b

def denormalize(T, T_max, T_min):
    a=2.0/(T_max - T_min)
    b=-(T_max + T_min)/(T_max - T_min)
    return (T-b)/a
################################################################################

Model=BasicUNet(in_channels=1, out_channels=1)

N_epochs=1000
lr=1.0e-04
eps=0.0
loss=torch.nn.MSELoss()
loss_list=[]

optimizer = torch.optim.Adam(Model.parameters(), lr=lr, eps=eps)

##################################################################################

for epoch in range(1, N_epochs + 1):
    optimizer.zero_grad()
    Y_pred = Model(X)
    loss_val = loss(Y, Y_pred)
    loss_val.backward()
    optimizer.step()
    loss_list.append(loss_val.detach().numpy())
    if epoch % int(N_epochs/10) == 0:
        print("%d epochs completed, loss_value = %f" % (epoch, loss_list[-1]), flush=True)

###################################################################################

torch.save(Model, 'UNet_vcb.pt')

###################################################################################