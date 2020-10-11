# -*- coding: utf-8 -*-
"""
Homework One Main Program
Created on Thu Sep 24 11:15:52 2020
@author: ancarey
"""

import torch as th
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import HomeworkOneNetwork as net
import HomeworkOneDataGeneration as data

#Set up training, validation, and testing data
training_data, training_labels = data.sample_points(10000)
validation_data, validation_labels = data.sample_points(2000)
testing_data, testing_labels = data.sample_points(2000)

#plot data to visualize
#data.plot(training_data, training_labels)
#data.plot(validation_data, validation_labels)
#data.plot(testing_data, testing_labels)

th.set_default_dtype(th.double)

#testing for optimal size of hidden layer
hidden_layer_size = [25]

for hidden_nodes in hidden_layer_size:
    
    network = net.Network(2, 2, hidden_nodes)
    loss = nn.CrossEntropyLoss()
    opt = optim.SGD(network.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    
    #training 
    epochs = [10, 100, 1000]
    training_labels = Variable(training_labels, requires_grad=False)
    
    for epoch in epochs:
        train_loss = []
        val_loss = []
        test_loss = []
        
        for e in range(epoch):
            network.train()
            opt.zero_grad()
            output = network(training_data)
            t_loss = loss(output, training_labels.long())
            
            t_loss.backward()
            opt.step()
            
            train_loss.append(t_loss.item())
            network.eval()
            output = network(validation_data)
            v_loss = loss(output, validation_labels.long())
            val_loss.append(v_loss.item())
            
            #print("hidden size:", hidden_nodes, "Epoch: ", epoch, "Train loss: ", np.mean(train_loss), "Valid loss: ", np.mean(val_loss))
        
        print("Training and Validating with ", epoch, " epochs and ", hidden_nodes, " nodes in the hidden layer.")
        print("Training loss : ", np.mean(train_loss))
        print("Validation loss: ", np.mean(val_loss))
        print("Testing with ", epoch, " epochs and ", hidden_nodes, " nodes in the hidden layer.")
        
        network.eval()
        output = network(testing_data)
        _, predictions = th.max(output, 1)
        accuracy = 0
        for i in range(len(predictions)):
            if(predictions[i] == testing_labels[i]):
                accuracy += 1 
        accuracy = accuracy / len(predictions)
        te_loss = loss(output, testing_labels.long())
        test_loss.append(te_loss.item())
            
        print("Testing loss: ", np.mean(test_loss))
        print("Testing accuracy: ", accuracy)
        
        plt.plot(train_loss, 'g-', label="train")
        plt.plot(val_loss, 'c-', label="validation")

        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Training and Validation with 25 nodes")
        plt.show()
    