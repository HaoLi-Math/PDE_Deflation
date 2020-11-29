#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:39:30 2020

@author: haoli
"""

import torch 
import numpy as np
from numpy import array, prod, sum, zeros, exp, max, log, sqrt
from torch.nn.functional import relu
from torch.nn import Softplus

torch.set_default_tensor_type('torch.cuda.DoubleTensor')
# torch.set_default_tensor_type('torch.DoubleTensor')
h = 0.001

left = 1
right = np.pi
scale = 1

def func(x):
    return (torch.sin(23*x) + torch.sin(137*x) + torch.sin(203*x)).view((x.shape[0],)).clone().detach()

# the point-wise residual: Input x is a batch of sampling points of d variables (tensor); Output is tensor vector
def res(net, tensor_x_batch):
    return net(tensor_x_batch) - func(tensor_x_batch)

# specify the domain type
def domain_shape():
    return 'cube'

# output the domain parameters
def domain_parameter(d):
    intervals = zeros((d,2))
    for i in range(d):
        intervals[i,:] = array([left,right/scale])
    return intervals

# If this is a time-dependent problem
def time_dependent_type():
    return 'none'

# output the time interval
def time_interval():
    return None
