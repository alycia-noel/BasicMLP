# -*- coding: utf-8 -*-
"""
Homework One Data Generation
Created on Thu Sep 24 11:22:16 2020

@author: ancarey
"""

import torch as th 
import numpy as np
import matplotlib.pyplot as plt

#return (x,y) where x of shape (n,2) is the numpy array of points
#and y is the (n) array of classes
def sample_points(n):
    #Uniform radius between 0 and 2
    radius = np.random.uniform(low=0, high=2, size=n).reshape(-1, 1)
    
    #uniform angle
    angle = np.random.uniform(low=0, high=2*np.pi, size=n).reshape(-1,1)
    
    x1 = radius*np.cos(angle)
    x2 = radius*np.sin(angle)
    y = (radius < 1).astype(int).reshape(-1)
    x = np.concatenate([x1, x2], axis=1)
    
    x = th.from_numpy(x)
    y = th.from_numpy(y)
    
    return x,y
    
def plot(data, labels):
    x_zero = []
    y_zero = []
    x_one = []
    y_one = []
    
    for i in range(len(labels)):
        if(labels[i].item() == 1):
            x_one.append(data[i].numpy()[0])
            y_one.append(data[i].numpy()[1])
        else:
            x_zero.append(data[i].numpy()[0])
            y_zero.append(data[i].numpy()[1])
            
    plt.plot(x_one, y_one, 'co', label=0)    
    plt.plot(x_zero, y_zero, 'mo', label=1)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("Testing Data")
    plt.legend(loc="upper right")
    plt.show()
    