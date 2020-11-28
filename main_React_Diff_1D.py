#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:56:03 2020

@author: haoli
"""

import torch
from torch import Tensor, optim
import numpy as  np
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_on_cube,\
    generate_learning_rates, generate_deflation_alpha, get_dxi
import Network_React_Diff_1D as network_file
# from selection_network_setting import selection_network
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy
import pickle
import time

from solution_React_Diff_1D import res1, res2, res, Bu_ft, h0, h1, domain_shape, domain_parameter, time_dependent_type, time_interval

# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
torch.set_default_tensor_type('torch.DoubleTensor')

########### Set parameters #############

method = 'B' # choose methods: B(basic), S(SelectNet), RS (reversed SelectNet)
d = 1  # dimension of problem
m = 100  # number of nodes in each layer of solution network
n_epoch = 50000  # number of outer iterations
N_inside_train = 1000 # number of trainning sampling points inside the domain in each epoch (batch size)
N_inside_test = 1000 # number of test sampling points inside the domain
N_pts_deflation = 1000 # number of deflation sampling points inside the domain
n_update_each_batch = 1 # number of iterations in each epoch (for the same batch of points)
lrseq = generate_learning_rates(-2,-4,n_epoch)
lambda_term = 100
p = 4
alpha = ones((n_epoch,)) # delfation parameter
alpha = generate_deflation_alpha(3,1,n_epoch)
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
scale = 10
right = right / scale
########### Network Setting #################
seperate_loss = False
flag_boundary_term_in_loss = True
activation = 'ReLU3'  # activation function for the solution net
boundary_control = 'none'  # if the solution net architecture satisfies the boundary condition automatically 
flag_preiteration_by_small_lr = False  # If pre iteration by small learning rates
lr_pre = 1e-4
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

flag_deflation = False

    
########### Interface parameters #############
flag_compute_loss_each_epoch = True
n_epoch_show_info = 100
flag_show_sn_info = False
flag_show_plot = True
flag_output_results = True
    
########### Depending parameters #############
net_A = network_file.network(d,m, activation_type = activation, boundary_control_type = boundary_control, initial_constant = initial_constant)
net_S = network_file.network(d,m, activation_type = activation, boundary_control_type = boundary_control, initial_constant = initial_constant)



if time_dependent_type == 'none':
    flag_initial_term_in_loss = False  # if loss function has the initial residual


if flag_boundary_term_in_loss or flag_initial_term_in_loss:
    flag_IBC_in_loss = True  # if loss function has the boundary/initial residual
    N_IBC_train = 0  # number of boundary and initial training points
else:
    flag_IBC_in_loss = False


## if the boundary is not enforeced by the NN, then put it in the loss
if flag_boundary_term_in_loss:
    if domain_shape == 'cube':
        if d == 1 and time_dependent_type == 'none':
            N_each_face_train = 1
        else:
            N_each_face_train = max([1,int(round(N_inside_train/2/d))]) # number of sampling points on each domain face when trainning
        N_boundary_train = 2*d*N_each_face_train
    elif domain_shape == 'sphere':
        if d == 1 and time_dependent_type == 'none':
            N_boundary_train = 2
        else:
            N_boundary_train = N_inside_train # number of sampling points on each domain face when trainning
    N_IBC_train = N_IBC_train + N_boundary_train
else:
    N_boundary_train = 0
    

if flag_initial_term_in_loss:          
    N_initial_train = max([1,int(round(N_inside_train/d))]) # number of sampling points on each domain face when trainning
    N_IBC_train = N_IBC_train + N_initial_train


#################### Start ######################
if seperate_loss:
    optimizer_A = optim.Adam(net_A.parameters(),lr=lrseq[0])
    if boundary_control == 'Neumann':
        optimizer_A.add_param_group({'params': net_A.c1})
        optimizer_A.add_param_group({'params': net_A.c2})
        
    optimizer_S = optim.Adam(net_S.parameters(),lr=lrseq[0])
    if boundary_control == 'Neumann':
        optimizer_S.add_param_group({'params': net_S.c1})
        optimizer_S.add_param_group({'params': net_S.c2})
else:
    optimizer = optim.Adam(net_A.parameters(),lr=lrseq[0])
    optimizer.add_param_group({'params': net_S.parameters()})
    if boundary_control == 'Neumann':
        optimizer.add_param_group({'params': net_A.c1, 'lr': 0.01})
        optimizer.add_param_group({'params': net_A.c2, 'lr': 0.01})
        optimizer.add_param_group({'params': net_S.c1, 'lr': 0.01})
        optimizer.add_param_group({'params': net_S.c2, 'lr': 0.01})

lossseq1 = zeros((n_epoch,))
resseq1 = zeros((n_epoch,))

lossseq2 = zeros((n_epoch,))
resseq2 = zeros((n_epoch,))


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
    if flag_IBC_in_loss == True:
        x2_train = generate_uniform_points_on_cube(domain_intervals,N_each_face_train)        
    
    
    tensor_x1_train = Tensor(x1_train)
    tensor_x1_train.requires_grad=False
    tensor_x1_test = Tensor(x1_test)
    tensor_x1_test.requires_grad=False
    tensor_x_deflation = Tensor(x_deflation)
    tensor_x_deflation.requires_grad=False
    if flag_boundary_term_in_loss == True:
        tensor_x2_train = Tensor(x2_train)
        tensor_x2_train.requires_grad=False

        
    ## Set learning rate
    if seperate_loss:
        for param_group in optimizer_A.param_groups:
            if flag_preiteration_by_small_lr == True and k == 0:
                param_group['lr'] = lr_pre
            else:
                param_group['lr'] = lrseq[k]
                
        for param_group in optimizer_S.param_groups:
            if flag_preiteration_by_small_lr == True and k == 0:
                param_group['lr'] = lr_pre
            else:
                param_group['lr'] = lrseq[k]
    else:
        for param_group in optimizer.param_groups:
            if flag_preiteration_by_small_lr == True and k == 0:
                param_group['lr'] = lr_pre
            else:
                param_group['lr'] = lrseq[k]

        
    if flag_preiteration_by_small_lr == True and k == 0:
        temp = n_update_each_batch_pre
    else:
        temp = n_update_each_batch
    for i_update in range(temp):
        if flag_deflation:
            deflation_A = 1/((torch.sum((net_A(tensor_x1_train)-A_s)**2))/tensor_x1_train.shape[0])**(p/2)
            deflation_S = 1/((torch.sum((net_S(tensor_x1_train)-S_s)**2))/tensor_x1_train.shape[0])**(p/2)
            deflation = deflation_A + deflation_S + alpha[k]
        if flag_compute_loss_each_epoch == True or i_update == 0:
            ## Compute the loss  
            loss1 = 1/N_inside_train*torch.sum(res1(net_A, net_S, tensor_x1_train)**2)
            


#        if i_update%10 == 0:
#            print("i_update = %d, loss = %6.3f, L2 error = %5.3f" %(i_update,loss.item(),evaluate_rel_l2_error(u_net, x1_train)))
        
        ## Update the network
        if seperate_loss:
            optimizer_A.zero_grad()
            optimizer_S.zero_grad()
            loss1.backward(retain_graph=not flag_compute_loss_each_epoch)
            optimizer_A.step()
            optimizer_S.step()
        
        if flag_compute_loss_each_epoch == True or i_update == 0:
            loss2 = 1/N_inside_train*torch.sum(res2(net_A, net_S, tensor_x1_train)**2)

            
        if seperate_loss:   
            optimizer_A.zero_grad()
            optimizer_S.zero_grad()
            loss2.backward(retain_graph=not flag_compute_loss_each_epoch)
            optimizer_A.step()
            optimizer_S.step()
        
        if not seperate_loss:
            loss = loss1 + loss2
            if flag_boundary_term_in_loss == True:
                loss = loss + lambda_term/N_IBC_train * (torch.sum((get_dxi(net_A, tensor_x2_train, 0) - 0)**2) + torch.sum((get_dxi(net_S, tensor_x2_train, 0) - 0)**2))
            if flag_deflation:
                loss = deflation*loss
            optimizer.zero_grad()
            loss.backward(retain_graph=not flag_compute_loss_each_epoch)
            optimizer.step()
        
    # Save loss and L2 error
    if seperate_loss:
        lossseq1[k] = loss1.item()
        resseq1[k] = np.sqrt(1/N_inside_train*torch.sum(res1(net_A, net_S, tensor_x1_test)**2).detach().numpy())
        lossseq2[k] = loss2.item()
        resseq2[k] = np.sqrt(1/N_inside_train*torch.sum(res2(net_A, net_S, tensor_x1_test)**2).detach().numpy())
    else:
        lossseq1[k] = loss.item()
        resseq1[k] = np.sqrt(1/N_inside_train*torch.sum(res1(net_A, net_S, tensor_x1_test)**2+res2(net_A, net_S, tensor_x1_test)**2).detach().numpy())
    ## Show information
    if k%n_epoch_show_info==0:
        if flag_show_plot == True:
            if i_update%10 == 0:
                # Plot the slice for xd
                clf()
                plt.plot(x_plot[:,0], net_A(torch.tensor(x_plot)).detach().numpy(),'r')
                plt.plot(x_plot[:,0], net_S(torch.tensor(x_plot)).detach().numpy(),'b')
                plt.legend(["Net_A", "Net_S"])
                plt.title("Epoch = "+str(k+1))
                show()
                pause(0.02)
        if flag_compute_loss_each_epoch:
            if seperate_loss:
                print("epoch = %d, deflation_term = %2.5f, loss1 = %2.5f, residual1 = %2.5f, loss2 = %2.5f, residual2 = %2.5f" %(k, deflation,loss1.item(),resseq1[k],loss2.item(),resseq2[k]), end='')
            else:
                if flag_deflation:
                    print("epoch = %d, deflation_term = %2.5f, loss = %2.5f, residual = %2.5f" %(k+1, deflation, loss1.item(), resseq1[k]), end='')
                else:
                    print("epoch = %d, loss = %2.5f, residual = %2.5f" %(k+1, loss1.item(), resseq1[k]), end='')
        else:
            print("epoch = %d" % k, end='')
        print("\n")
    
    k = k + 1

# compute plot at x_plot
net_A_plot = net_A.predict(x_plot)
net_S_plot = net_S.predict(x_plot)

# Save net for deflation
localtime = time.localtime(time.time())
time_text = str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)
# torch.save(net_A.state_dict(),'networkpara_A_'+time_text+'.pkl')
# torch.save(net_S.state_dict(),'networkpara_S_'+time_text+'.pkl')

# Output results
if flag_output_results == True:
    # compute the plot 
    N_plot = 201

    net_A_plot = net_A.predict(x_plot)
    net_S_plot = net_S.predict(x_plot)
    
    #save the data
    main_file_name = 'main_React_Diff'
    data = {'main_file_name':main_file_name,\
                                'ID':time_text,\
                                'N_inside_train':N_inside_train,\
                                'N_plot':N_plot,\
                                'activation':activation,\
                                'boundary_control':boundary_control,\
                                'd':d,\
                                'domain_shape':domain_shape,\
                                'flag_preiteration_by_small_lr':flag_preiteration_by_small_lr,\
                                'flag_reset_select_net_each_epoch':flag_reset_select_net_each_epoch,\
                                'selectnet_initial_constant':selectnet_initial_constant,\
                                'h_Du_t':h_Du_t,\
                                'initial_constant':initial_constant,\
                                'lambda_term':lambda_term,\
                                'lossseq1':lossseq1,\
                                'lossseq2':lossseq2,\
                                'lr_pre':lr_pre,\
                                'lrseq':lrseq,\
                                'm':m,\
                                'method':method,\
                                'n_epoch':n_epoch,\
                                'n_update_each_batch':n_update_each_batch,\
                                'n_update_each_batch_pre':n_update_each_batch_pre,\
                                'resseq1':resseq2,\
                                'resseq2':resseq2,\
                                'time_dependent_type':time_dependent_type,\
                                'net_A_plot':net_A_plot,\
                                'net_S_plot':net_S_plot,\
                                'x_plot':x_plot,\
                                }
    filename = 'React_Diff_d_'+str(d)+'_'+method+'_'+time_text+'.data'
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

# empty the GPU memory
# torch.cuda.empty_cache()
