#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 12:59:54 2022

@author: ananyakapoor
"""

import numpy as np
import control
import matplotlib.pyplot as plt
import torch
import torch.nn as nn # All the neural network modules
import torch.optim as optim
import scipy


dt = 0.01
tau_train = 1
timepoints = np.arange(0,1,dt/tau_train)
y_des_train = np.sin(np.arange(0, 1, dt/tau_train) * 5)
# y_des_train = np.zeros(timepoints.shape)
# y_des_train[int(len(y_des_train) / 2.0) :] = 0.5
goal = y_des_train[-1]

hidden_units = 2
n_input = 1
Wr=np.random.normal(0,1,hidden_units**2)
Wr.shape=(hidden_units,hidden_units)

# B = np.zeros((hidden_units,1))
# B[0] = 1
B = np.ones((hidden_units, n_input))

W_y = np.random.normal(0,1,hidden_units)
W_y.shape=(1,hidden_units)
# W_y = np.array([1, 0])

Q = np.eye(hidden_units)
R = np.eye(n_input)

D = 0

K, S, E = control.dlqr(Wr, B, Q, R)


Atilde = Wr- np.matmul(B, K)

sys_lqr = control.StateSpace(Atilde, B, W_y, D, dt = True)
ss_results_lqr = control.step_response(sys_lqr)

K_constant = (1/control.dcgain(sys_lqr))
Btilde = B* K_constant*goal
scaled_sys_lqr = control.StateSpace(Atilde, Btilde, W_y, D, dt = True)  
# sys_results_scaled = control.step_response(scaled_sys_lqr, T = 100,return_x = True)
(time, y_base, state_base)  = control.step_response(scaled_sys_lqr, T = int(1/(dt/tau_train))-1,return_x = True)

plt.figure()
plt.plot(y_base)
plt.show()


class DMP_RNN(nn.Module):
    def __init__(self, hidden_size_forcing, hidden_size_trans,
                 num_layers_forcing, num_layers_trans,
                 learning_rate, num_epoch, sequence_length, batch_size, output_size, input_size_forcing, input_size_trans):
        super(DMP_RNN, self).__init__()
        self.hidden_size_forcing = hidden_size_forcing
        self.hidden_size_trans = hidden_size_trans
        
        self.num_layers_forcing = num_layers_forcing
        self.num_layers_trans = num_layers_trans
        
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.output_size = output_size
        self.input_size_forcing = input_size_forcing
        self.input_size_trans = input_size_trans
        
        self.rnn_forcing = nn.RNN(self.input_size_forcing, self.hidden_size_forcing, self.num_layers_forcing,batch_first=True)
        self.fc_forcing=nn.Linear(self.hidden_size_forcing, self.output_size, bias = False)
        # self.fc_to_output = nn.Linear(1, self.output_size, bias=False) #f(x) and y_base are inputs and will be mapped to yhat
        
        self.rnn_trans = nn.RNN(self.input_size_trans, self.hidden_size_trans, self.num_layers_trans,batch_first=True, bias=False)
        self.fc_trans=nn.Linear(self.input_size_trans, self.output_size, bias = False)
        # self.fc_to_output = nn.Linear(self.hidden_size_trans,self.output_size, bias = False)
    
    def forward_forcing(self, x):
        h0 = torch.randn(self.num_layers_forcing,self.batch_size, self.hidden_size_forcing)
        output_forcing, _ = self.rnn_forcing(x, h0)
        pred_forcing = self.fc_forcing(output_forcing)
        return pred_forcing
 

    def forward_trans(self, x):
        h0 = torch.randn(self.num_layers_trans,self.batch_size, self.hidden_size_trans)
        output_trans, _ = self.rnn_trans(x, h0)
        pred_trans = self.fc_trans(output_trans)
        return pred_trans

hidden_size_forcing = 100
hidden_size_trans = hidden_units
num_layers_forcing = 2
num_layers_trans = 1
learning_rate = 0.001
num_epoch = 5000
sequence_length = 10
batch_size = int(int(1/(dt/tau_train))/sequence_length)
output_size = 1
input_size_forcing = 1
input_size_trans = 2


DMP_System = DMP_RNN(hidden_size_forcing, hidden_size_trans,
             num_layers_forcing, num_layers_trans,
             learning_rate, num_epoch, sequence_length, batch_size, output_size, input_size_forcing, input_size_trans)

criterion=nn.MSELoss()
optimizer=optim.Adam(DMP_System.parameters(),lr=learning_rate)

# Generate the canonical system
alpha = 1
tvals = np.linspace(0,1,int(1/(dt/tau_train)))
xvals = np.exp(-1*alpha*tvals)
xvals = np.array(xvals)

DMP_System.rnn_trans.weight_hh_l0 = nn.Parameter(torch.tensor(Atilde).float())
DMP_System.rnn_trans.weight_hh_l0.requires_grad = False

DMP_System.fc_trans.weight = nn.Parameter(torch.tensor(W_y).float())
DMP_System.fc_trans.weight.requires_grad = False

training_loss=[]
position_predictions = np.zeros((timepoints.shape[0], num_epoch))

for epoch in range(num_epoch):
    # Training 
    DMP_System.batch_size = batch_size
    inputs = torch.tensor(xvals).float()
    inputs=inputs.reshape(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)
    # forcing_preds=DMP_System.forward_forcing(inputs)*inputs*(goal_train-y_des_train[0])
    forcing_preds=DMP_System.forward_forcing(inputs)
    # forcing_preds=inputs # We are removing the forcing function from the system 
    LQR_input = float(Btilde[0])*torch.ones((DMP_System.batch_size, DMP_System.sequence_length, 1))
    trans_input = torch.stack((forcing_preds, LQR_input), axis= 2).squeeze()
    position_preds = DMP_System.forward_trans(trans_input)
    
    targets=torch.tensor(y_des_train).float()
    targets=targets.reshape(batch_size, sequence_length, output_size)
    
    optimizer.zero_grad()
    loss = criterion(position_preds, targets)
    training_loss.append(loss.item())
  
    position_preds_arr=position_preds.detach().numpy()
    position_preds_arr.shape=(batch_size*sequence_length,)
    
    position_predictions[:,epoch]=position_preds_arr
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        plt.clf();
        plt.ion()
        plt.title(str("Computed Position, ")+"Epoch {}".format(epoch))
        plt.plot(y_des_train,'r-',linewidth=1,label='Target Position')
        plt.plot(position_predictions[:,epoch],linewidth=1,label='Predictions')
        plt.legend()
        plt.draw();
        plt.pause(0.05);
        
        print(epoch, loss.item()) 

tau_test = 1
timepoints = np.arange(0,1,dt/tau_test)
y_des_test = np.sin(np.arange(0, 1, dt/tau_test) * 5)

goal_test = 5
y_des_test[-1]=goal_test


# Testing 

Btilde_test = B* K_constant*goal_test
scaled_sys_lqr_test = control.StateSpace(Atilde, Btilde_test, W_y, D, dt = True)  
# sys_results_scaled = control.step_response(scaled_sys_lqr, T = 100,return_x = True)
(time_test, y_base_test, state_base_test)  = control.step_response(scaled_sys_lqr_test, T = int(1/(dt/tau_test))-1,return_x = True)

plt.figure()
plt.plot(y_base_test)
plt.show()

DMP_System.batch_size = batch_size
inputs = torch.tensor(xvals).float()
inputs=inputs.reshape(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)
# forcing_preds=DMP_System.forward_forcing(inputs)*inputs*(goal_train-y_des_train[0])
forcing_preds=DMP_System.forward_forcing(inputs)
forcing_preds*=inputs
# forcing_preds=inputs # We are removing the forcing function from the system 
LQR_input = float(Btilde_test[0])*torch.ones((DMP_System.batch_size, DMP_System.sequence_length, 1))
trans_input = torch.stack((forcing_preds, LQR_input), axis= 2).squeeze()
position_preds = DMP_System.forward_trans(trans_input)


position_preds_arr = position_preds.detach().numpy()
position_preds_arr.shape = (batch_size*sequence_length,)

plt.figure()
plt.title("Testing Performance")
plt.plot(position_preds_arr, label ="Predictions")
plt.plot(y_des_test, 'r', label = "Testing Trajectory")
plt.show()

