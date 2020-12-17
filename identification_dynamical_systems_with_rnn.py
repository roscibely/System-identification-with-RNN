# -*- coding: utf-8 -*-
'''Identification dynamical systems with RNN.ipynb

# *Using RNN for System Identification*
'''


## Necessary packages
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv         
from control import lqr
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.optimizers import RMSprop
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

"""## Discretization with backward Euler"""
def backward_Euler_model(A,B,C,initial_state,input_sequence, time_steps,sampling_period):
    #function simulates the state-space model using the backward Euler method
    I=np.identity(A.shape[0]) 
    Ad=inv(I-sampling_period*A)
    Bd=Ad*sampling_period*B
    states=np.zeros(shape=(A.shape[0],time_steps+1))
    output=np.zeros(shape=(C.shape[0],time_steps+1))
    control_law=np.zeros(shape=(B.shape[0],time_steps+1))
    for i in range(0,time_steps):
       if i==0:
           states[:,[i]]=initial_state
           u=-states[:,[i]]*input_sequence
           output[:,[i]]=C*initial_state
           x=Ad*initial_state+u*Bd
       else:
           states[:,[i]]=x
           u=-states[:,[i]]*input_sequence
           output[:,[i]]=C*x
           x=Ad*x+u*Bd
    states[:,[-1]]=x
    output[:,[-1]]=C*x
    control_law[:,[-1]] =-input_sequence*x
    return states, output, control_law

## Ball beam system implementation
# System parameters 
Lbeam= 0.4255
r_arm=0.0254
R=0.0127
g=9.81
m= 0.064
Jb= 4.129*10**(-6)
Kl=1.5286
tau= 0.0248
const = (m*r_arm*g*R**2)/(Lbeam*(m*R**2 +Jb))
# State space dynamics
A = np.matrix([[0, 1, 0, 0],
              [0, 0, const ,0],
              [0, 0, 0, 1],
              [0, 0, 0, -1/tau]])
B = np.matrix([[0],
              [0],
              [0],
              [61.6371]])
C=np.matrix([[1, 0, 0 ,0]])
D=0
# Number of time-samples
time=20    
# Sampling time for the discretization                           
sampling=0.01
# An initial state for simulation                         
X0=np.array([[0.2125, 0, -0.9774, 0 ]]).T
"""## LQR control"""
# LQR cost matrices
Q = np.array([[0.1, 0, 0 ,0], [0, 0.1, 0 ,0], [1, 1, 0.1 ,1], [0, 0, 0 ,0.1]], dtype=np.float)
R = np.array([[0.01]], dtype=np.float)
# Obtain LQR controlelr gain and cost-to-go matrix
Kn, Pn, EE = lqr(A, B, Q, R)

"""## Discrete system simulation """
state,output, U=backward_Euler_model(A,B,C,X0,Kn, time ,sampling)    
plt.plot(output[0,:], 'k')
plt.xlabel('Discrete time instant-k')
plt.ylabel('Ball Position')
plt.title('System with LQR control response')

"""## System identification with RNNs"""
#####################################Create the train data
uk_train=Kn                  #Define an input sequence for the simulation
x0_train=np.random.rand(4,1)        #Define an initial state for simulation
                                    #Here we simulate the dynamics
state,output_train, U=backward_Euler_model(A,B,C,x0_train,uk_train, time ,sampling)  
uk_train=U  
output_train=output_train.T
output_train=np.concatenate((output_train, np.zeros(shape=(2,1))), axis=0)
                                    #This is the output data used for training
output_train=np.reshape(output_train,(1,output_train.shape[0],1))
tmp_train=np.concatenate((uk_train, np.zeros(shape=(uk_train.shape[0],1))), axis=1)
tmp_train=np.concatenate((x0_train.T,tmp_train.T), axis=0)
                                    #This is the input data used for training
trainX=np.reshape(tmp_train, (1,tmp_train.shape[0],tmp_train.shape[1]))

#####################################Create the validation data
uk_validate=Kn
x0_validate=np.random.rand(4,1)     #New random initial condition
# create a new ouput sequence by simulating the system 
state_validate,output_validate,UU=backward_Euler_model(A,B,C,x0_validate,uk_validate, time ,sampling)    
output_validate=output_validate.T
output_validate=np.concatenate((output_validate, np.zeros(shape=(2,1))), axis=0)
                                    #This is the output data used for validation
output_validate=np.reshape(output_validate,(1,output_validate.shape[0],1))
uk_validate=UU
tmp_validate=np.concatenate((uk_validate, np.zeros(shape=(uk_validate.shape[0],1))), axis=1)
tmp_validate=np.concatenate((x0_validate.T,tmp_validate.T), axis=0)
validateX=np.reshape(tmp_validate, (1,tmp_validate.shape[0],tmp_validate.shape[1]))

########################################Create the test data
uk_test=Kn 
x0_test=np.random.rand(4,1) #New random initial condition
                            #Create a new ouput sequence by simulating the system 
state_test,output_test, UT=backward_Euler_model(A,B,C,x0_test,uk_test, time ,sampling)    
output_test=output_test.T
output_test=np.concatenate((output_test, np.zeros(shape=(2,1))), axis=0)
                            #This is the output data used for test
output_test=np.reshape(output_test,(1,output_test.shape[0],1))
uk_test=UT
tmp_test=np.concatenate((uk_test, np.zeros(shape=(uk_test.shape[0],1))), axis=1)
tmp_test=np.concatenate((x0_test.T,tmp_test.T), axis=0)
testX=np.reshape(tmp_test, (1,tmp_test.shape[0],tmp_test.shape[1]))

### Network models

"""1. LSTM"""
modelLSTM=Sequential()
modelLSTM.add(LSTM(32, input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences=True))
modelLSTM.add(TimeDistributed(Dense(1)))  
modelLSTM.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mse'])
filepath="\\python_files\\system_identification\\modelsLSTM\\weights-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]
historyLSTM=modelLSTM.fit(trainX, output_train , epochs=2000, batch_size=1, callbacks=callbacks_list, validation_data=(validateX,output_validate), verbose=2)
testPredictLSTM = modelLSTM.predict(testX)

""" 2. RNN model"""
modelSimpleRNN=Sequential()
modelSimpleRNN.add(SimpleRNN(32, input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences=True))
modelSimpleRNN.add(TimeDistributed(Dense(1))) 
modelSimpleRNN.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mse'])
filepath="\\python_files\\system_identification\\modelsSimpleRNN\\weights-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]
historySimpleRNN=modelSimpleRNN.fit(trainX, output_train , epochs=2000, batch_size=1, callbacks=callbacks_list, validation_data=(validateX,output_validate), verbose=2)
testPredictSimpleRNN = modelSimpleRNN.predict(testX)

"""3. GRU model"""
modelGRU=Sequential()
modelGRU.add(GRU(32, input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences=True))
modelGRU.add(TimeDistributed(Dense(1))) 
modelGRU.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mse'])
filepath="\\python_files\\system_identification\\modelsGRU\\weights-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]
historyGRU=modelGRU.fit(trainX, output_train , epochs=2000, batch_size=1, callbacks=callbacks_list, validation_data=(validateX,output_validate), verbose=2)
testPredictGRU = modelGRU.predict(testX)

"""Figures """
plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 12}) 
plt.plot(testPredictGRU[0,:,0], 'm-.', linewidth=3.0, label='Predicted output GRU')
plt.plot(testPredictLSTM[0,:,0], 'c--', linewidth=3.0, label='Predicted output LSTM')
plt.plot(testPredictSimpleRNN[0,:,0], 'g:+', linewidth=3.0, label='Predicted output simple RNN')
plt.plot(output_test[0,:],'r-', linewidth=3.0, label='Real output')
plt.xlabel('Discrete time steps')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

lossGRU=historyGRU.history['loss']
val_lossGRU=historyGRU.history['val_loss']
lossLSTM=historyLSTM.history['loss']
val_lossLSTM=historyLSTM.history['val_loss']
lossSimpleRNN=historySimpleRNN.history['loss']
val_lossSimpleRNN=historySimpleRNN.history['val_loss']
epochs=range(1,len(lossGRU)+1)
plt.figure()
plt.plot(epochs, lossGRU, 'm-.', label='Training loss GRU')
plt.plot(epochs, val_lossGRU,'mo', label='Validation loss GRU')
plt.plot(epochs, lossLSTM,'c--', label='Training loss LSTM')
plt.plot(epochs, val_lossLSTM,'cx', label='Validation loss LSTM')
plt.plot(epochs, val_lossSimpleRNN,'g+', label='Validation loss simple RNN')
plt.plot(epochs, lossSimpleRNN,'g:', label='Training loss simple RNN')
plt.ylim(0,0.075)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.show()