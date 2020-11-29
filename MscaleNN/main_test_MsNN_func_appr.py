#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:22:27 2020

@author: haoli
"""

import torch
from torch import Tensor, optim
import numpy as  np
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_on_cube,\
    generate_learning_rates, generate_deflation_alpha, get_dxi
import MscaleNetwork as network_file
# from selection_network_setting import selection_network
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy
import pickle
import time
from solution_HighFrequency import res, domain_shape, domain_parameter, time_dependent_type, time_interval, func

torch.set_default_tensor_type('torch.cuda.DoubleTensor')
# torch.set_default_tensor_type('torch.DoubleTensor')

########### Set parameters #############

d = 1  # dimension of problem
m = 500  # number of nodes in each layer of solution network
K = 100  # scale number of for MscaleNN
n_epoch = 50000  # number of outer iterations
N_inside_train = 1000 # number of trainning sampling points inside the domain in each epoch (batch size)
N_inside_test = 1000 # number of test sampling points inside the domain
N_pts_deflation = 1000 # number of deflation sampling points inside the domain
n_update_each_batch = 1 # number of iterations in each epoch (for the same batch of points)
# lrseq = generate_learning_rates(-2,-4,n_epoch)
lambda_term = 100


########### Network Setting #################


activation = 'sReLU'  # activation function for the solution net
boundary_control = 'none'  # if the solution net architecture satisfies the boundary condition automatically 
flag_preiteration_by_small_lr = True  # If pre iteration by small learning rates
lr_pre = 1e-5
n_update_each_batch_pre = 100
h_Du_t = 0.01  # time length for computing the first derivative of t by finite difference (for the hyperbolic equations)
flag_reset_select_net_each_epoch = False  # if reset selection net for each outer iteration
selectnet_initial_constant = 1  # if selectnet is initialized as constant one
initial_constant = 'none'

########### Problem parameters   #############

time_dependent_type = time_dependent_type()   ## If this is a time-dependent problem
domain_shape = domain_shape()  ## the shape of domain 
if domain_shape == 'cube':  
    domain_intervals = domain_parameter(d)
elif domain_shape == 'sphere':
    R = domain_parameter(d)
    
if not time_dependent_type == 'none':    
    time_interval = time_interval()
    T0 = time_interval[0]
    T1 = time_interval[1]

########### Interface parameters #############
flag_compute_loss_each_epoch = True
n_epoch_show_info = 100
flag_show_sn_info = False
flag_show_plot = False
flag_output_results = True
    
########### Depending parameters #############
net = network_file.network(d,m,K, activation_type = activation, boundary_control_type = boundary_control, initial_constant = initial_constant)

optimizer = optim.Adam(net.parameters(),lr=lr_pre)
lossseq = zeros((n_epoch,))
resseq = zeros((n_epoch,))


N_plot = 1001
x_plot = np.zeros((N_plot,d))
x_plot[:,0] = np.linspace(domain_intervals[0,0],domain_intervals[0,1],N_plot)

# Training
k = 0
while k < n_epoch:
    ## generate training and testing data (the shape is (N,d)) or (N,d+1) 
    ## label 1 is for the points inside the domain, 2 is for those on the bondary or at the initial time
    x1_train = generate_uniform_points_in_cube(domain_intervals, N_inside_train)
    x1_test = generate_uniform_points_in_cube(domain_intervals, N_inside_test)
    x_deflation = generate_uniform_points_in_cube(domain_intervals, N_pts_deflation)

    
    tensor_x1_train = Tensor(x1_train)
    tensor_x1_train.requires_grad=False
    tensor_x1_test = Tensor(x1_test)
    tensor_x1_test.requires_grad=False
    tensor_x_deflation = Tensor(x_deflation)
    tensor_x_deflation.requires_grad=False
    for param_group in optimizer.param_groups:
        if flag_preiteration_by_small_lr == True:
            param_group['lr'] = lr_pre
        else:
            param_group['lr'] = lrseq[k]
    
    if flag_preiteration_by_small_lr == True and k == 0:
        temp = n_update_each_batch_pre
    else:
        temp = n_update_each_batch
    for i_update in range(temp):
        loss = 1/N_inside_train*torch.sum(res(net, tensor_x1_train)**2)
        optimizer.zero_grad()
        loss.backward(retain_graph=not flag_compute_loss_each_epoch)
        optimizer.step()
    lossseq[k] = loss.item()
    resseq[k] = np.sqrt(1/N_inside_train*torch.sum(res(net, tensor_x1_test)**2).detach().numpy())
    ## Show information
    if k%n_epoch_show_info==0:
        if flag_show_plot == True:
            if i_update%10 == 0:
                # Plot the slice for xd
                clf()
                plt.plot(x_plot[:,0], net(torch.tensor(x_plot)).detach().numpy(),'r')
                plt.plot(x_plot[:,0], func(torch.tensor(x_plot)).detach().numpy(),'b')
                plt.legend(["Net", "func"])
                plt.title("Epoch = "+str(k+1))
                show()
                pause(0.02)
        if flag_compute_loss_each_epoch:
            print("epoch = %d, loss = %2.5f, residual = %2.5f" %(k+1, loss.item(), resseq[k]))
        else:
            print("epoch = %d" % k)
        print("\n")
    
    k = k + 1