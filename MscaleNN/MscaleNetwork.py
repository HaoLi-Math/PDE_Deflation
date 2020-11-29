#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:17:34 2020

@author: haoli
"""

# build the neural network to approximate the solution
import torch
import numpy
from numpy import pi
from torch import tanh, squeeze, sin, cos, sigmoid, autograd, sqrt
from torch.nn.functional import relu


# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
torch.set_default_tensor_type('torch.DoubleTensor')

class network(torch.nn.Module):
    def __init__(self, d, m, K = 50, activation_type = 'ReLU', boundary_control_type = 'none', initial_constant = 'none'):
        super(network, self).__init__()
        self.layer1 = torch.nn.Linear(d,2*m)
        self.layer2 = torch.nn.Linear(2*m,m)
        self.layer3 = torch.nn.Linear(m,m)
        self.layer3 = torch.nn.Linear(m,m)
        self.layer3 = torch.nn.Linear(m,m)
        self.layer4 = torch.nn.Linear(m,1)
        if activation_type == 'ReLU3':
            self.activation = lambda x: relu(x**3)
        elif activation_type == 'ReLU':
            self.activation = lambda x: relu(x)
        elif activation_type == 'sReLU':
            self.activation = lambda x: relu(x)*relu(1-x)
        elif activation_type == 'sigmoid':
            self.activation = lambda x: sigmoid(x)
        elif activation_type == 'tanh':
            self.activation = lambda x: tanh(x)
        elif activation_type == 'sin':
            self.activation = lambda x: sin(x)
        self.boundary_control_type = boundary_control_type
        if boundary_control_type == 'none':
            self.boundary_enforeced = False
        else:
            self.boundary_enforeced = True
        if boundary_control_type == 'Neumann':
            self.c1 = torch.tensor(0.)
            self.c1.requires_grad = True
            self.c2 = torch.tensor(0.)
            self.c2.requires_grad = True
        if not initial_constant == 'none':
            torch.nn.init.constant_(self.layer3.bias, initial_constant)
        self.K = torch.tensor([i//(m//K) for i in range(2*m)])+1
        self.K.requires_grad = False

    def forward(self, tensor_x_batch):
        y = self.layer1(tensor_x_batch)
        y = self.K * (y - self.layer1.bias) + self.layer1.bias
        y = self.layer2(self.activation(y))
        y = self.layer3(self.activation(y))
        y = self.layer4(self.activation(y))
        if self.boundary_control_type == 'none':
            return y.squeeze(1)
        elif self.boundary_control_type == 'Dirichlet':
            return y.squeeze(1)*tensor_x_batch[:,0]*(tensor_x_batch[:,0]-1)+10**(0.5)*tensor_x_batch[:,0]
        
