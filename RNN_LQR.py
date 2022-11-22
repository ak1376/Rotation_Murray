#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 07:13:16 2022

@author: ananyakapoor
"""

import numpy as np
import scipy
import control
import matplotlib.pyplot as plt
from numpy.random import default_rng
import torch
import torch.nn as nn # All the neural network modules
import torch.optim as optim

dt = 0.001
timepoints = np.arange(0,1,dt)
tau_train = 1
y_des_train = np.sin(np.arange(0, 1, dt/tau_train) * 5)
# y_des_train = np.zeros(timepoints.shape)
# y_des_train[int(len(y_des_train) / 2.0) :] = 0.5
goal = y_des_train[-1]

# First we will generate a recurrent weight matrix that is random normal
hidden_units = 2
Wr=np.random.normal(5,10,hidden_units**2)
Wr.shape=(hidden_units,hidden_units)

# B = np.zeros((hidden_units,1))
# B[0] = 1
B = np.ones((hidden_units, 2))

W_y = np.random.normal(1,0.05,hidden_units)
W_y.shape=(1,hidden_units)
# W_y = np.array([1, 0])

Q = 100*np.eye(hidden_units)
R = 0.001*np.eye(2)

D = 0

K, S, E = control.lqr(Wr, B, Q, R)

# r_initial = np.zeros((hidden_units, 1))

# r = r_initial
# u = np.matmul(-K, r) 

# for t in np.arange(int(1/(dt/tau_train))): 
#     r = np.matmul(Wr,r) + np.matmul(B, u) 

Atilde_lqr = Wr-np.matmul(B,K)
sys_lqr = control.StateSpace(Atilde_lqr, B, W_y, D)
ss_results_lqr = control.step_response(sys_lqr, T=timepoints, squeeze=True)

K_constant = (1/control.dcgain(sys_lqr))

scaled_sys_lqr = control.StateSpace(Atilde_lqr, B* K_constant*goal, W_y, D)  
scaled_ss_results_lqr = control.step_response(scaled_sys_lqr, T=timepoints)

y_base = np.squeeze(scaled_ss_results_lqr.outputs)
y_base=y_base[0,:]

plt.figure()
plt.plot(y_base)
plt.title("LQR Trajectory")
plt.xlabel("Timepoint")
plt.ylabel("Position")
plt.show()

# Forcing RNN

class DMP_RNN(nn.Module):
    def __init__(self, hidden_size_forcing, 
                 num_layers_forcing,
                 learning_rate, num_epoch, sequence_length, batch_size, output_size, input_size_forcing):
        super(DMP_RNN, self).__init__()
        self.hidden_size_forcing = hidden_size_forcing
        self.num_layers_forcing = num_layers_forcing
                
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.output_size = output_size
        self.input_size_forcing = input_size_forcing
        
        self.rnn_forcing = nn.RNN(self.input_size_forcing, self.hidden_size_forcing, self.num_layers_forcing,batch_first=True)
        self.fc_forcing=nn.Linear(self.hidden_size_forcing, self.output_size)
        self.fc_to_output = nn.Linear(1, self.output_size, bias=False) #f(x) and y_base are inputs and will be mapped to yhat
        
    def forward_forcing(self, x):
        h0 = torch.randn(self.num_layers_forcing,self.batch_size, self.hidden_size_forcing)
        output_forcing, _ = self.rnn_forcing(x, h0)
        pred_forcing = self.fc_forcing(output_forcing)
        return pred_forcing
    
    def forward_to_yhat(self, fval, y_base):
        # h0 = torch.randn(self.num_layers_forcing,self.batch_size, self.hidden_size_forcing)
        # output_forcing, _ = self.rnn_forcing(x, h0)
        pred_value = self.fc_to_output(fval)+y_base
        return pred_value

# Generate the canonical system

alpha = 1
# goal = 500

tvals_train = np.linspace(0, 1, int(1/(dt/tau_train)))


xvals=[]
x=1

for t in np.arange(tvals_train.shape[0]):
    xvals.append(x)
    x+=-alpha*x*(dt/tau_train)

xvals = np.array(xvals)

hidden_size_forcing = 100
num_layers_forcing = 1
learning_rate = 0.01
num_epoch = 5000
sequence_length = 10
batch_size = int(int(1/(dt/tau_train))/sequence_length)
output_size = 1
input_size_forcing = 1


DMP_System = DMP_RNN(hidden_size_forcing, 
             num_layers_forcing,
             learning_rate, num_epoch, sequence_length, batch_size, output_size, input_size_forcing)


criterion=nn.MSELoss()
optimizer=optim.Adam(DMP_System.parameters(),lr=learning_rate)

loss_list=[]
for epoch in range(num_epoch):
    
    # Training 
    DMP_System.batch_size = batch_size
    inputs = torch.tensor(xvals).float()
    inputs=inputs.reshape(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)
    forcing_preds=DMP_System.forward_forcing(inputs)
    # forcing_preds = torch.tensor(np.zeros((DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)))
    # forcing_preds_arr = np.zeros((DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing))
    # forcing_preds_arr.shape=(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)
    # forcing_preds = torch.tensor(forcing_preds_arr).float()
    
    
    y_base.shape=(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)
    y_base_torch = torch.tensor(y_base).float()
    # inputs_final = np.concatenate((y_base, forcing_preds_arr),axis=2)
    # inputs_final=torch.tensor(inputs_final).float()
    
    yhat_torch = DMP_System.forward_to_yhat(forcing_preds, y_base_torch)
    yhat = yhat_torch.detach().numpy()
    yhat.shape=(DMP_System.batch_size*DMP_System.sequence_length,)
    
    targets=torch.tensor(y_des_train).float()
    targets=targets.reshape(batch_size, sequence_length, output_size)

    optimizer.zero_grad()
    loss = criterion(yhat_torch, targets)
    loss_list.append(loss.item())
    
    loss.backward()
    optimizer.step()
    
    if epoch%10 ==0:
        # plt.clf();
        # plt.ion()
        # plt.title(str("Computed Position, ")+"Epoch {}".format(epoch))
        # plt.plot(y_des_train,'r-',linewidth=1,label='Target Position')
        # plt.plot(yhat,linewidth=1,label='Predictions')
        # plt.legend()
        # plt.draw();
        # plt.pause(0.05);
        
        print("Epoch Number: "+str(epoch)+", Loss Value: "+str(loss.item()))

forcing_preds_arr = forcing_preds.detach().numpy()
forcing_preds_arr.shape = (batch_size*sequence_length,)
# Testing on a new trajectory


tau_test = 1
y_des_test = np.sin(np.arange(0, 1, dt/tau_test) * 5)
# goal = 4
# y_des_test[-1] = goal
# y_des_test = np.zeros(int(1/(dt/tau_test)))
# y_des_test[int(len(y_des_test) / 2.0) :] = 2
goal = y_des_test[-1]

tvals_test = np.linspace(0, 1, int(1/(dt/tau_test)))


xvals_test=[]
x=1

timepoints_test = np.arange(0,1,dt/tau_test)


for t in np.arange(tvals_test.shape[0]):
    xvals_test.append(x)
    x+=-alpha*x*(dt/tau_test)

xvals_test = np.array(xvals_test)

scaled_sys_lqr = control.StateSpace(Atilde_lqr, B* K_constant*goal, W_y, D)  
scaled_ss_results_lqr = control.step_response(scaled_sys_lqr, T=timepoints_test)

y_base_test = np.squeeze(scaled_ss_results_lqr.outputs)
y_base_test=y_base_test[0,:]

batch_size_test = int(int(1/(dt/tau_test))/sequence_length)

# Testing 
DMP_System.batch_size = batch_size_test
inputs = torch.tensor(xvals_test).float()

inputs=inputs.reshape(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)
forcing_preds=DMP_System.forward_forcing(inputs)

y_base_test.shape=(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)
y_base_test_torch = torch.tensor(y_base_test).float()
# inputs_final = np.concatenate((y_base, forcing_preds_arr),axis=2)
# inputs_final=torch.tensor(inputs_final).float()

yhat_torch = DMP_System.forward_to_yhat(forcing_preds, y_base_test_torch)

yhat_test = yhat_torch.detach().numpy()
yhat_test.shape=(DMP_System.batch_size*DMP_System.sequence_length,)

plt.figure()
plt.title("Spatial Generalizability")
plt.plot(y_des_test, 'r', label = "Testing Trajectory")
plt.plot(yhat_test, label="Predictions")
plt.legend()
plt.show()












