#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:57:41 2020

@author: haoli
"""

# the solution is mulit-D sin function: u(x,t) = exp(-t)(x1^2-1)*(x2^2-1)*...*(xd^2-1)
import torch 
import numpy as np
from numpy import array, prod, sum, zeros, exp, max, log, sqrt
from torch.nn.functional import relu
from torch.nn import Softplus

# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
torch.set_default_tensor_type('torch.DoubleTensor')
h = 0.001

########### Solution Parameter ############
rho = np.random.uniform(0,1);
mu = np.random.uniform(2/(1+rho)-1,2)
D = np.random.uniform(0.001,1/mu*(np.sqrt(2/(1+rho))-1)**2)

gamma = 0.5
delta_A = np.random.normal(1)
delta_S = np.random.normal(1)

rho = 0.01
D = 0.1
mu = 1

# Steady State solution
A_s = 1+rho
S_s = (1+rho)**(-2)

left = 0
right = 10

# the point-wise Du: Input x is a sampling point of d variables ; Output is a numpy vector which means the result of Du(x))
def Du(model,x_batch):
    ei = zeros(x_batch.shape)
    ei[:,0] = 1
    s = (model.predict(x_batch+h*ei)-2*model.predict(x_batch)+model.predict(x_batch-h*ei))/h/h
    s = s - scale*scale*model.predict(x_batch)**2 + scale*scale*scale*x_batch[:,0]
    return s

# the point-wise residual: Input x is a batch of sampling points of d variables (tensor); Output is tensor vector
def res1(net_A, net_S, tensor_x_batch):
    s = torch.zeros((tensor_x_batch.shape[0],))
    for i in range(tensor_x_batch.shape[1]):
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:,i] = 1
        # if i == 0:
        #     s = - (net_A(tensor_x_batch+h*ei)-net_A(tensor_x_batch-h*ei))/h/2
        # else:
        #     s = s + D*(net_A(tensor_x_batch+h*ei)-2*net_A(tensor_x_batch)+net_A(tensor_x_batch-h*ei))/h/h
        s = s + D*(net_A(tensor_x_batch+h*ei)-2*net_A(tensor_x_batch)+net_A(tensor_x_batch-h*ei))/h/h
    s = s + net_S(tensor_x_batch)*net_A(tensor_x_batch)**2 -net_A(tensor_x_batch) + rho
    return s.reshape(tensor_x_batch.shape)

def res2(net_A, net_S, tensor_x_batch):
    s = torch.zeros((tensor_x_batch.shape[0],))
    for i in range(tensor_x_batch.shape[1]):
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:,0] = 1
        s = s + (net_S(tensor_x_batch+h*ei)-2*net_S(tensor_x_batch)+net_S(tensor_x_batch-h*ei))/h/h
    s = s + mu*(1 - net_S(tensor_x_batch)*net_A(tensor_x_batch)**2)
    return s.reshape(tensor_x_batch.shape)

def res(net_A, net_S, tensor_x_batch):
    return res1(net_A, net_S, tensor_x_batch) + res2(net_A, net_S, tensor_x_batch)

# define the right hand function for numpy array (N sampling points of d variables)
def f(x_batch):
    f = zeros((x_batch.shape[0],))
    return f

# the point-wise Bu for tensor (N sampling points of d variables)
def Bu_ft(model,tensor_x_batch):
    return model(tensor_x_batch)


# the point-wise h0 numpy array (N sampling points of d variables)
def h0(x_batch):
    return None

# the point-wise h1 for numpy array (N sampling points of d variables)
def h1(x_batch):
    return None


# specify the domain type
def domain_shape():
    return 'cube'

# output the domain parameters
def domain_parameter(d):
    intervals = zeros((d,2))
    for i in range(d):
        intervals[i,:] = array([left,right])
    return intervals

# If this is a time-dependent problem
def time_dependent_type():
    return 'none'

# output the time interval
def time_interval():
    return None

