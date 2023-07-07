#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 18:15:46 2022

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

# Step 2: Build the Augmented System that uses the forcing function

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
        self.fc_forcing=nn.Linear(self.hidden_size_forcing, self.output_size, bias = True)
        # self.fc_to_output = nn.Linear(1, self.output_size, bias=False) #f(x) and y_base are inputs and will be mapped to yhat
        
        # self.rnn_trans = nn.RNN(self.input_size_trans, self.hidden_size_trans, self.num_layers_trans,batch_first=True, bias=False)
        # self.fc_trans=nn.Linear(self.hidden_size_trans, self.output_size, bias = False)
        self.fc_trans=nn.Linear(self.input_size_trans, self.output_size, bias = False)
        # self.fc_to_output = nn.Linear(self.hidden_size_trans,self.output_size, bias = False)
    
    def forward_forcing(self, x):
        h0 = torch.randn(self.num_layers_forcing,self.batch_size, self.hidden_size_forcing)
        output_forcing, _ = self.rnn_forcing(x, h0)
        pred_forcing = self.fc_forcing(output_forcing)
        return pred_forcing
    
    # def forward_trans(self, x):
    #     h0 = torch.randn(self.num_layers_trans,self.batch_size, self.hidden_size_trans)
    #     output_trans, _ = self.rnn_trans(x, h0)
    #     pred_trans = self.fc_trans(output_trans)
    #     return pred_trans
 
hidden_size_forcing = 100
hidden_size_trans = hidden_units
# hidden_size_trans = 50
num_layers_forcing = 1
num_layers_trans = 1
learning_rate = 0.01
num_epoch = 5000
sequence_length = 5
batch_size = int(int(1/(dt/tau_train))/sequence_length)
output_size = 1
input_size_forcing = 1
input_size_trans = 1

       
DMP_System = DMP_RNN(hidden_size_forcing, hidden_size_trans,
             num_layers_forcing, num_layers_trans,
             learning_rate, num_epoch, sequence_length, batch_size, output_size, input_size_forcing, input_size_trans)

criterion=nn.MSELoss()
optimizer=optim.Adam(DMP_System.parameters(),lr=learning_rate)
# optimizer = torch.optim.SGD(DMP_System.parameters(), lr=learning_rate)
# STep 3: Train the model

# Generate the canonical system
alpha = 5
# tvals_train = np.linspace(0, 1, int(1/(dt/tau_train)))
# xvals=[]
# x=1
# for t in np.arange(tvals_train.shape[0]):
#     xvals.append(x)
#     x+=-alpha*x*(dt/tau_train)
    
tvals = np.linspace(0,1,int(1/(dt/tau_train)))
xvals = np.exp(-1*alpha*tvals)
xvals = np.array(xvals)

loss_list=[]
forcing_preds_list = np.zeros((int(1/(dt/tau_train)), num_epoch))

# def vdp1(t, state_base):
#     rprime_base = scaled_sys_lqr.dynamics(timepoints[t], state_base[:,t])
#     rprime_base.shape = (DMP_System.hidden_size_trans, 1)
#     rprime_base = torch.tensor(rprime_base).float()
#     forcing_input_to_ssm_val = forcing_input_to_ssm[:,t]
#     forcing_input_to_ssm_val=forcing_input_to_ssm_val.reshape(DMP_System.hidden_size_trans,1)
#     rprime_base+= forcing_input_to_ssm_val
#     rprime = rprime_base
#     return rprime

# DMP_System.fc_trans.weight = nn.Parameter(torch.tensor([[1]]).float())
# DMP_System.fc_trans.weight.requires_grad = False

Atilde = torch.tensor(Atilde).float()
Btilde = torch.tensor(Btilde).float()
W_y = torch.tensor(W_y).float()
D = torch.tensor(D).float()


for epoch in range(num_epoch):
    # Training
    DMP_System.batch_size = batch_size
    inputs = torch.tensor(xvals).float()
    inputs=inputs.reshape(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)
    # forcing_preds=DMP_System.forward_forcing(inputs)
    forcing_preds=inputs*DMP_System.forward_forcing(inputs)
    # forcing_preds=forcing_preds*inputs*(goal-y_des_train[0])
    forcing_preds_arr=forcing_preds.detach().numpy()
    forcing_preds_arr.shape = (DMP_System.batch_size*DMP_System.sequence_length,)
    # plt.plot(forcing_preds_arr)
    forcing_preds_list[:,epoch] = forcing_preds_arr
    forcing_input_to_ssm = DMP_System.fc_trans(forcing_preds)
    forcing_input_to_ssm = forcing_preds
    
    # y_list = DMP_System.forward_trans(forcing_input_to_ssm)
    # yhat_arr = torch.zeros((timepoints.shape[0],1))
    # DMP_System.fc_to_output.weight = nn.Parameter(torch.tensor(W_y).float())
    # DMP_System.fc_to_output.weight.requires_grad = False
    
    # # We want to create a new state space model with the input from the forcing function
    # u = forcing_input_to_ssm.detach().numpy()
    # # u.shape = (batch_size*sequence_length,DMP_System.hidden_size_trans)
    # u.shape = (100,1)
    # system = (Atilde, Btilde, W_y, D, dt/tau_train)
    # t_in = np.linspace(0,1,int(1/(dt/tau_train)))
    # tout, yout, xout = scipy.signal.dlsim(system, u, t=t_in)
    
    #r_initial = torch.tensor([[0],[0]]).float()
    # r_initial = np.zeros((DMP_System.hidden_size_trans,1))
    # r_state = r_initial
    # C = torch.tensor(W_y).float()
    # y_list = torch.empty(0, 1) 
    # r_state_vec = torch.zeros((100,2))
    # counter = 0
    # y_list = torch.empty(0, 1)

    # for b in range(DMP_System.batch_size):
    #     forcing_input_to_ssm_batch = forcing_input_to_ssm[b,:,:]
    #     for t in range(0, DMP_System.sequence_length):
    #         forcing_val = forcing_input_to_ssm_batch[t,:]
    #         # forcing_val = forcing_val.reshape(DMP_System.hidden_size_trans, 1)
    #         forcing_val_arr = forcing_val.detach().numpy()
    #         # forcing_val = torch.zeros((hidden_size_trans,1))
    #         # forcing_val.requires_grad = True
    #         rprime = np.matmul(Atilde, r_state) + B*goal*K_constant +forcing_val_arr
    #         r_state += rprime*(dt/tau_train)
    #         r_state_torch = torch.tensor(r_state).float()
    #         y = DMP_System.fc_to_output(r_state_torch.T)
    #         # y = np.matmul(W_y, r_state)
    #         # y=DMP_System.fc_to_output(r_state.T)
    #         # y_list.append(float(y))
    #         # y_list = torch.cat((y_list, y), axis =0)
    #         y_list = torch.vstack((y_list, y))

    # INFLEXIBLE CODE BELOW
    
    forcing_input_to_ssm = forcing_input_to_ssm.reshape(batch_size*sequence_length, 1)
    # forcing_val = forcing_input_to_ssm.detach().numpy()
    u_dt = torch.ones((timepoints.shape[0],1))
    u_dt = u_dt.reshape(batch_size*sequence_length, 1)
    # u_dt.shape = (batch_size*sequence_length, 1)
    
    r_initial = torch.zeros((1,DMP_System.hidden_size_trans))
    rout = torch.zeros((timepoints.shape[0], DMP_System.hidden_size_trans), requires_grad=False)
    yout = torch.zeros((timepoints.shape[0], DMP_System.output_size), requires_grad = False)
    
    rout[0,:] = r_initial
    for t in range(0, timepoints.shape[0] - 1):
        rout[t+1, :] = (torch.matmul(Atilde, rout[t, :]) +
                        torch.matmul(Btilde, u_dt[t, :]) +
                        forcing_input_to_ssm[t,:])
        yout[t, :] = torch.matmul(W_y, rout[t, :])

    
    
    # forcing_val = forcing_input_to_ssm.detach().numpy()
    # forcing_val.shape = (batch_size*sequence_length, 1)
    # u_dt = np.ones((timepoints.shape[0],1))
    # u_dt.shape = (batch_size*sequence_length, 1)
    
    # # Simulate the system
    # r_initial = np.zeros((1,DMP_System.hidden_size_trans))
    # rout = np.zeros((timepoints.shape[0], DMP_System.hidden_size_trans))
    # yout = np.zeros((timepoints.shape[0], DMP_System.output_size))
    # rout[0,:] = r_initial
    # for t in range(0, timepoints.shape[0] - 1):
    #     rout[t+1, :] = (np.dot(Atilde, rout[t, :]) +
    #                     np.dot(Btilde, u_dt[t, :]) +
    #                     forcing_val[t,:])
    #     yout[t, :] = np.tanh(np.dot(W_y, rout[t, :]) +
    #                   np.dot(D, u_dt[t, :]))

    # Last point
    yout[timepoints.shape[0]-1, :] = (torch.matmul(W_y, rout[timepoints.shape[0]-1, :]))

    #tout, y, xout = scipy.signal.dlsim(system, u, t = timepoints)
    # yout.shape = (batch_size, sequence_length, output_size)
    # y_list = yout.clone()
    # y_list = torch.tensor(yout).float()
    # y_list.requires_grad = True
    
    targets=torch.tensor(y_des_train).float()
    targets=targets.reshape(batch_size, sequence_length, output_size)
    yout = yout.reshape(batch_size, sequence_length, output_size)
    optimizer.zero_grad()
    loss = criterion(yout, targets)
    loss_list.append(loss.item())
    
    loss.backward()
    optimizer.step()
    
    yhat_arr = yout.detach().numpy()
    yhat_arr.shape = (batch_size*sequence_length,)
    
    
    if epoch%10 ==0:
        # plt.clf();
        # plt.ion()
        # plt.title(str("Computed Position, ")+"Epoch {}".format(epoch))
        # plt.plot(y_des_train,'r-',linewidth=1,label='Target Position')
        # plt.plot(yhat_arr,linewidth=1,label='Predictions')
        # plt.legend()
        # plt.draw();
        # plt.pause(0.05);
        
        print("Epoch Number: "+str(epoch)+", Loss Value: "+str(loss.item()))



plt.figure()
plt.plot(y_des_train,'r',label="Training Trajectory")
plt.plot(yhat_arr, label ="Predictions")
plt.legend()
plt.show()

# TESTING 

# Basic parameters

tau_test = 1
timepoints_test = np.arange(0,1,dt/tau_test)
y_des_test = np.sin(np.arange(0, 1, dt/tau_test) * 5)
# y_des_test = np.zeros(timepoints_test.shape)
# y_des_test[int(len(y_des_test) / 2.0) :] = goal_test
goal_test = -10
y_des_test[-1]=goal_test


# Learning the LQR 

Btilde_test = B* K_constant*goal_test
scaled_sys_lqr_test = control.StateSpace(Atilde, Btilde_test, W_y, D, dt = True)  
# sys_results_scaled = control.step_response(scaled_sys_lqr, T = 100,return_x = True)
(time_test, y_base_test, state_base_test)  = control.step_response(scaled_sys_lqr_test, T = int(1/(dt/tau_test))-1,return_x = True)

plt.figure()
plt.plot(y_base_test)
plt.show()

# Generate the canonical system
tvals_test = np.linspace(0,1,int(1/(dt/tau_test)))
xvals_test = np.exp(-1*alpha*tvals_test)
xvals_test = np.array(xvals_test)

# Validate the model 

batch_size_test = int(int(1/(dt/tau_test))/sequence_length)
DMP_System.batch_size = batch_size_test
inputs_test = torch.tensor(xvals_test).float()
inputs_test=inputs_test.reshape(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size_forcing)

forcing_preds_test=inputs_test*DMP_System.forward_forcing(inputs_test)
# forcing_preds_arr=forcing_preds.detach().numpy()
# forcing_preds_arr.shape = (DMP_System.batch_size*DMP_System.sequence_length,)
# plt.plot(forcing_preds_arr)
# forcing_preds_list[:,epoch] = forcing_preds_arr
forcing_input_to_ssm_test = DMP_System.fc_trans(forcing_preds_test)
forcing_input_to_ssm_test = forcing_preds_test

forcing_input_to_ssm_test = forcing_input_to_ssm_test.reshape(batch_size_test*sequence_length, 1)
# forcing_val = forcing_input_to_ssm.detach().numpy()
u_dt_test = torch.ones((timepoints_test.shape[0],1))
u_dt_test = u_dt_test.reshape(batch_size_test*sequence_length, 1)
# u_dt.shape = (batch_size*sequence_length, 1)
Btilde_test = torch.tensor(Btilde_test).float()
r_initial = torch.zeros((1,DMP_System.hidden_size_trans))
rout_test = torch.zeros((timepoints_test.shape[0], DMP_System.hidden_size_trans), requires_grad=False)
yout_test = torch.zeros((timepoints_test.shape[0], DMP_System.output_size), requires_grad = False)

rout_test[0,:] = r_initial
for t in range(0, timepoints_test.shape[0] - 1):
    rout_test[t+1, :] = (torch.matmul(Atilde, rout_test[t, :]) +
                    torch.matmul(Btilde_test, u_dt_test[t, :]) +
                    forcing_input_to_ssm_test[t,:])
    yout_test[t, :] = torch.matmul(W_y, rout_test[t, :])

yout_test[timepoints_test.shape[0]-1, :] = (torch.matmul(W_y, rout_test[timepoints_test.shape[0]-1, :]))

yout_test_arr=yout_test.detach().numpy()
yout_test_arr.shape = (batch_size_test*sequence_length,1)

plt.figure()
plt.plot(y_des_test, 'r', label ='Testing Trajectory')
plt.plot(yout_test_arr, label = 'Predictions')
plt.legend()
plt.show()
