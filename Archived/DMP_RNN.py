#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:35:39 2022

@author: ananyakapoor
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn # All the neural network modules
import torch.optim as optim

torch.manual_seed(295)

# We will train a series of RNNs in order to achieve a particular trajectory (let's start off with position first but normally this would be velocity). Our pipeline will look as follows: 
# time -> nonlinear transformation of time "x" -> premotor population activity u -> forcing function f -> primariy motor population activity "r" -> readout position

# First let's define some base parameters
dt = 0.01 # Our timestep size
tau_train = 1 # Our time constant value
# y_des_train= np.zeros(int(1/dt))
# y_des_train[int(len(y_des_train) / 2.0) :] = 0.5
y_des_train = np.sin(np.arange(0, 1, dt/tau_train) * 5) # Our training trajectory
goal_train = y_des_train[-1] # The goal location that we want our system to go to
alpha=1 # The alpha value for the canonical system

# First let's plot the trajectory 
tvals_train = np.linspace(0, 1, int(1/dt))
plt.figure()
plt.title("Training Trajectory")
plt.plot(y_des_train, color = 'blue')
plt.show()

# First we want to transform our time values into x(t)

xvals=[]
x=1

for t in np.arange(tvals_train.shape[0]):
    xvals.append(x)
    x+=-alpha*x*(dt/tau_train)
    

# Now let's plot the canonical system
plt.figure()
plt.title("Training Canonical System")
plt.plot(xvals)
plt.ylabel("X Value")
plt.xlabel("Time Step Number")
plt.show()

# Now let's test our model on a testing trajectory
    
tau_test = 0.5
  
# y_des_test = np.zeros(int(1/dt))
# y_des_test[int(len(y_des_test) / 2.0) :] = 0.5
y_des_test = np.sin(np.arange(0, 1, dt/tau_test) * 5) # Our training trajectory
goal_test = y_des_test[-1] # The goal location that we want our system to go to
y_des_test[-1]=goal_test
alpha=1 # The alpha value for the canonical system

# First let's plot the trajectory 
tvals_test = np.linspace(0, 1, int(1/(dt/tau_test)))
plt.figure()
plt.title("Testing Trajectory")
plt.plot(y_des_test, color = 'green')
plt.show()

# First we want to transform our time values into x(t)

xvals_test=[]
x=1

for t in np.arange(tvals_test.shape[0]):
    xvals_test.append(x)
    x+=-alpha*x*(dt/tau_test)
   

# Now let's plot the canonical system
plt.figure()
plt.title("Testing Canonical System")
plt.plot(xvals_test)
plt.xlabel("X Value")
plt.ylabel("Time Step Number")
plt.show()    

# Forcing function hyperparameters
hidden_size_forcing = 100
num_layers_forcing=1

# Transformation system hyperparameters
hidden_size_trans=100
num_layers_trans=1

num_epoch =5000
learning_rate = 0.01
sequence_length = 5 
batch_size = int(int(1/dt)/sequence_length)
output_size = 1
input_size = 1

batch_size_test= int(int(1/(dt/tau_test))/sequence_length)

class DMP_RNN(nn.Module):
    def __init__(self, hidden_size_forcing, 
                 num_layers_forcing,
                 hidden_size_trans, num_layers_trans,
                 learning_rate, num_epoch, sequence_length, batch_size, output_size, input_size):
        super(DMP_RNN, self).__init__()
        self.hidden_size_forcing = hidden_size_forcing
        self.num_layers_forcing = num_layers_forcing
        self.hidden_size_trans = hidden_size_trans
        self.num_layers_trans = num_layers_trans
                
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.output_size = output_size
        self.input_size = input_size 
        
        self.rnn_forcing = nn.RNN(self.input_size, self.hidden_size_forcing, self.num_layers_forcing,batch_first=True)
        self.fc_forcing=nn.Linear(self.hidden_size_forcing, self.output_size)
        
        self.rnn_trans = nn.RNN(self.input_size, self.hidden_size_trans, self.num_layers_trans,batch_first=True)
        self.fc_trans=nn.Linear(self.hidden_size_trans, self.output_size)
    
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
        

DMP_System = DMP_RNN(hidden_size_forcing,
             num_layers_forcing,
             hidden_size_trans, num_layers_trans, learning_rate, num_epoch, sequence_length, batch_size, output_size, input_size)


criterion=nn.MSELoss()
optimizer=optim.Adam(DMP_System.parameters(),lr=learning_rate)

position_predictions = np.zeros((batch_size*sequence_length, num_epoch))
position_predictions_test = np.zeros((int(batch_size_test)*sequence_length, num_epoch))
training_loss=[]
testing_loss=[]
for epoch in range(num_epoch):
    
    # Training 
    DMP_System.batch_size = batch_size
    inputs = torch.tensor(xvals).float()
    inputs=inputs.reshape(DMP_System.batch_size, DMP_System.sequence_length, DMP_System.input_size)
    # forcing_preds=DMP_System.forward_forcing(inputs)*inputs*(goal_train-y_des_train[0])
    forcing_preds=DMP_System.forward_forcing(inputs)
    # forcing_preds=inputs # We are removing the forcing function from the system 
    position_preds = DMP_System.forward_trans(forcing_preds)
    
    targets=torch.tensor(y_des_train).float()
    targets=targets.reshape(batch_size, sequence_length, output_size)
    
    optimizer.zero_grad()
    loss = criterion(position_preds, targets)
    training_loss.append(loss.item())
  
    position_preds_arr=position_preds.detach().numpy()
    position_preds_arr.shape=(batch_size*sequence_length,)
    
    position_predictions[:,epoch]=position_preds_arr
    
    # Testing
    
    with torch.no_grad():
        DMP_System.batch_size = batch_size_test
        inputs_test = torch.tensor(xvals_test).float()
        inputs_test=inputs_test.reshape(batch_size_test, DMP_System.sequence_length, DMP_System.input_size)
        forcing_preds_test=DMP_System.forward_forcing(inputs_test)
        position_preds_test = DMP_System.forward_trans(forcing_preds_test)
        
        targets_test=torch.tensor(y_des_test).float()
        
        
        targets_test=targets_test.reshape(batch_size_test, sequence_length, output_size)
        
        loss_test = criterion(position_preds_test, targets_test)
        testing_loss.append(loss_test.item())
            
        position_preds_arr_test=position_preds_test.detach().numpy()
        position_preds_arr_test.shape=(batch_size_test*sequence_length,)
        
        position_predictions_test[:,epoch]=position_preds_arr_test

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
    
    loss.backward()
    optimizer.step()
  

# Now let's see how well the model performed

# Training 
plt.figure()
plt.title("Model Performance on the Training Trajectory")
plt.plot(y_des_train, 'r-', label = "Training Target Trajectory")
plt.plot(position_predictions[:,-1], color = 'blue', label = "Predicted Training Trajectory")  
plt.legend()
plt.show()  

# Testing

plt.figure()
plt.title("Model Performance on the Testing Trajectory")
plt.plot(y_des_test, 'r-', label = "Testing Target Trajectory")
plt.plot(position_predictions_test[:,-1], color = 'green', label = "Predicted Testing Trajectory")  
plt.legend()
plt.show()  


# Now let's plot the training and validation losses by epoch
plt.figure()
plt.title("Loss Curves")
plt.plot(training_loss, label = 'Training Loss')
plt.plot(testing_loss, color='green', label = 'Testing Loss')
plt.xlabel("Epoch Number")
plt.ylabel("Loss (Mean Squared Error)")
plt.legend()
plt.show()


# In the DMP paper, if we remove the forcing function we should get simple
# convergence to the goal state. However, if we remove the forcing layer in the
# implementation above we still get nonlinear convergence to the goal state.
# This is because the transformation system layer is "picking up the slack" and
# learning the nonlinear relationship. 

# In fact, we need to rework the transformation system because it is optimized
# to achieve the goal state. This will not get us the spatial generalization to
# new goal states. Instead we need to cleverly define our transformation system
# so that the goal state is an attractor state of the system. Let's first work
# with a toy example where tau*y'' = alpha*(beta*(g-y)-y'). This is NOT 
# biologically plausible but this toy example should prove that when we have no
# forcing function we will converge to the goal state 

   
    
    
    
    
    
    
    
    
    
    
    
    

