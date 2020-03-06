#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:23:08 2020

@author: shashank
"""
import numpy as np
import h5py
import random
import matplotlib.pyplot as plt 

epochs = 50
batch_size = 1000
l2 = 0.02
e = 10**-10

DATA_FNAME = "mnist_traindata.hdf5"
with h5py.File(DATA_FNAME, 'r') as hf:
    data_x = hf['xdata'][:]
    data_y = hf['ydata'][:]


DATA_FNAME = "mnist_testdata.hdf5"
with h5py.File(DATA_FNAME, 'r') as hf:
    test_x = hf['xdata'][:]
    test_y = hf['ydata'][:]
    
activation_configurations = ["relu","tanh"]
learning_rate_configurations = [0.002,0.005,0.02]

def ReLU(x):
    return np.maximum(0,x),(x > 0) * 1

def Tanh(x):
    return np.tanh(x), 1-np.tanh(x)**2

def Softmax(x):
    return np.exp(x - x.max(axis=0))/(np.exp(x -  x.max(axis=0))+e).sum(axis = 0)

def cross_entropy(y, y_hat):
    return np.multiply(y,np.log(y_hat + e)).sum(axis=0).mean()

def saturate(x):
    y = np.zeros([10])
    y[np.where(x == np.amax(x))[0][0]] = 1
    return y

class NN():
    def __init__(self,learning_rate,act1="relu",act2="relu"):
        self.W1 = np.random.randn(200,784)*0.01
        self.W2 = np.random.randn(100,200)*0.01
        self.W3 = np.random.randn(10,100)*0.01
        self.b1 = np.zeros(200,)
        self.b2 = np.zeros(100,)
        self.b3 = np.zeros(10,)
        
        self.learning_rate = learning_rate
        self.act1 = act1
        assert act1 in ["relu", "tanh"]
        self.act2 = act2
        assert act2 in ["relu", "tanh"]

        
    def NNForward(self,x,mode):
        if mode == "train":
            s = batch_size
        elif mode == "test":
            s = x.shape[1]
        
        self.a0 = x
        s1 = self.W1@x + np.array([self.b1]*s).T
        if self.act1 == "relu":
            self.a1,self.a1_dot = ReLU(s1)
        elif self.act1 == "tanh":
            self.a1,self.a1_dot = Tanh(s1)
        s2 = self.W2@self.a1 + np.array([self.b2]*s).T
        if self.act2 == "relu":
            self.a2,self.a2_dot = ReLU(s2)
        elif self.act2 == "tanh":
            self.a2,self.a2_dot = Tanh(s2)
        s3 = self.W3@self.a2 + np.array([self.b3]*s).T
        self.y = Softmax(s3)
        return self.y

    def NNBackward(self, delta3):
        delta2 = np.multiply(self.a2_dot, (self.W3.T@delta3))
        delta1 = np.multiply(self.a1_dot, (self.W2.T@delta2))
        self.W3 = self.W3 - (self.learning_rate/batch_size)*(delta3@(self.a2.T) + 2*l2*self.W3)
        self.b3 = self.b3 - (self.learning_rate/batch_size)*(delta3.sum(axis = 1) + 2*l2*self.b3)
        self.W2 = self.W2 - (self.learning_rate/batch_size)*(delta2@(self.a1.T) + 2*l2*self.W2)
        self.b2 = self.b2 - (self.learning_rate/batch_size)*(delta2.sum(axis = 1) + 2*l2*self.b2) 
        self.W1 = self.W1 - (self.learning_rate/batch_size)*(delta1@(self.a0.T) + 2*l2*self.W1)
        self.b1 = self.b1 - (self.learning_rate/batch_size)*(delta1.sum(axis = 1) + 2*l2*self.b1)         
    
def train(nn,data_x,data_y,epochs):
    x_train,x_valid = data_x[0:50000],data_x[50000:]
    y_train,y_valid = data_y[0:50000],data_y[50000:]
    
    x_valid = x_valid.T
    y_valid = y_valid.T
    t=[]
    v=[]
    for k in range(epochs):
        #shuffle data
        idx = random.sample(range(50000), 50000)
        x_train = x_train[idx]
        y_train = y_train[idx]
        
        if not k%20 and k>0:
            print("Learning rate changed {} -> {}".format(nn.learning_rate,nn.learning_rate/2))
            nn.learning_rate = nn.learning_rate/2
        cost = 0
        for j in range(int(len(x_train)/batch_size)):
            x_batch = x_train[j*batch_size:(j+1)*batch_size].T
            y_batch = y_train[j*batch_size:(j+1)*batch_size].T
            y_hat = nn.NNForward(x_batch,"train")
            
            cost += cross_entropy(y_batch, y_hat)
            delta = y_hat-y_batch
            nn.NNBackward(delta)
    
        #training acuracy
        y_pred = nn.NNForward(x_train.T,"test")
        at = 0
        for i,(y,y_hat) in enumerate(zip(y_train,y_pred.T)):
            if (y==saturate(y_hat)).all():
                at += 1
        
        #validation
        y_pred = nn.NNForward(x_valid,"test")
        ct = 0
        for i,(y,y_hat) in enumerate(zip(y_valid.T,y_pred.T)):
            if (y==saturate(y_hat)).all():
                ct += 1
        t_acu = float(at/len(y_train))
        v_acu = float(ct/len(y_valid.T))
        t.append(t_acu)
        v.append(v_acu)
        print("Epoch {0}: average cost: {1:.4f} training accuracy: {2:.4f} validation accuracy: {3:.4f}".\
              format(k,float(cost/int(len(x_train)/batch_size)),\
                  t_acu,v_acu))
    return t,v

def test(test_x,test_y):    
    y_pred = nn.NNForward(test_x.T,"test")
    ct = 0
    for i,(y,y_hat) in enumerate(zip(test_y,y_pred.T)):
        if (y==saturate(y_hat)).all():
            ct += 1
    print("Test: accuracy: {}".format(ct/len(test_y)))
    
def trainFinal(nn,data_x,data_y,epochs):
    x_train = data_x
    y_train = data_y
    
    t=[]
    for k in range(epochs):
        #shuffle data
        idx = random.sample(range(50000), 50000)
        x_train = x_train[idx]
        y_train = y_train[idx]
        
        if not k%20 and k>0:
            print("Learning rate changed {} -> {}".format(nn.learning_rate,nn.learning_rate/2))
            nn.learning_rate = nn.learning_rate/2
        cost = 0
        for j in range(int(len(x_train)/batch_size)):
            x_batch = x_train[j*batch_size:(j+1)*batch_size].T
            y_batch = y_train[j*batch_size:(j+1)*batch_size].T
            y_hat = nn.NNForward(x_batch,"train")
            
            cost += cross_entropy(y_batch, y_hat)
            delta = y_hat-y_batch
            nn.NNBackward(delta)
    
        #training acuracy
        y_pred = nn.NNForward(x_train.T,"test")
        at = 0
        for i,(y,y_hat) in enumerate(zip(y_train,y_pred.T)):
            if (y==saturate(y_hat)).all():
                at += 1
        
        t_acu = float(at/len(y_train))
        t.append(t_acu)
        print("Epoch {0}: average cost: {1:.4f} training accuracy: {2:.4f}".\
              format(k,float(cost/int(len(x_train)/batch_size)),t_acu))
    return t

def testConfigs(activation_configurations,learning_rate_configurations):    
    fig = plt.figure()
    k=0
    for i,act in enumerate(activation_configurations):
        for j,lr in enumerate(learning_rate_configurations):
            k+=1
            nn = NN(lr,act,act)
            print("{0}: Training for activation {1} and LR {2:.6f}".format(k,act,lr))
            t,v = train(nn,data_x,data_y,epochs)
            ax = fig.add_subplot(len(activation_configurations), len(learning_rate_configurations), k)
            ax.plot([i for i in range(epochs)],t,'b',[i for i in range(epochs)],v,'r',label=["training", "validation"])
            ax.set_title("LR:{0:.4f} {1}".format(lr,act),fontsize=7)
            ax.set_ylim(0.1,1)
            ax.set_xlabel('epoch', fontsize=5)
            ax.set_ylabel('accuracy', fontsize=5)
            # ax.ti
            ax.tick_params(axis='both', labelsize=5)
    fig.tight_layout(pad=2.0)
    fig.savefig('fig.png', format='png', dpi=1200)

testConfigs(activation_configurations,learning_rate_configurations)

# best config is relu and LR = 0.01
nn = NN(0.01,"relu","relu")
t = trainFinal(nn,data_x,data_y,epochs)
plt.plot([i for i in range(epochs)],t)
test(test_x,test_y)

# Report:
# Network configuration: 
#   Input Layer = (784,)
#   Dense Layer = (200,)
#   Relu
#   Dense Layer = (100,)
#   Relu
#   Output Layer  = (10,)

# Batch size = 1000
# learning rates = [0.001,0.005,0.01]
# final test accuracy = 0.978