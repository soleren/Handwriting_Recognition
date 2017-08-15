#-*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import os
class NeuralNet:
    def __init__(self, inodes, hnodes, onodes, lrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        # set learning rate
        self.lr = lrate

        # set link weights
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        # print(self.wih,end='\n\n')
        # print(self.who,end='\n\n')

        # set activation function

        # self.activation_function = lambda x: 1/(1+np.exp(-x))
    def activation_function(self,x):
        # if (1/(1+np.exp(-x)) < 0.01):
        #     return 0.01
        # def ReLU(x):
        #     return x * (x > 0)
        #
        # def dReLU(x):
        #     return 1. * (x > 0)
        return 1/(1+np.exp(-x))

    def train(self,input_list,target_list):
        # convert inputs list to 2d array
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # output layer error is the (target ­ actual)
        output_errors = targets - final_outputs
        print(output_errors.mean())
        # hidden layer error is the output_errors, split by weights,recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def work(self,input_list):
        # convert inputs list to 2d array
        inputs = np.array(input_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

if __name__ == "__main__":

    input_nodes = 40000
    hidden_nodes = 400
    output_nodes =  5
    learning_rate = 0.001
    n = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # im0 = Image.open(os.getcwd()+"\\img\\00.JPG")
    # r0 = np.dot(np.array(im0), [0.299, 0.587, 0.114]).ravel()
    # scaled0 = (r0 / 255 * 0.98) + 0.1

    im1 = Image.open(os.getcwd()+"\\img\\11.JPG")
    r1 = np.dot(np.array(im1), [0.299, 0.587, 0.114]).ravel()
    scaled1 = (r1 / 255 * 0.98) + 0.1

    im2 = Image.open(os.getcwd()+"\\img\\22.JPG")
    r2 = np.dot(np.array(im2), [0.299, 0.587, 0.114]).ravel()
    scaled2 = (r2 / 255 * 0.98) + 0.1

    im3 = Image.open(os.getcwd()+"\\img\\33.JPG")
    r3 = np.dot(np.array(im3), [0.299, 0.587, 0.114]).ravel()
    scaled3 = (r3 / 255 * 0.98) + 0.1

    im4 = Image.open(os.getcwd()+"\\img\\44.JPG")
    r4 = np.dot(np.array(im4),[0.299, 0.587, 0.114]).ravel()
    scaled4 = (r4 / 255 * 0.98) + 0.1

    im11 = Image.open(os.getcwd()+"\\img\\111.JPG")
    r11 = np.dot(np.array(im11),[0.299, 0.587, 0.114]).ravel()
    scaled11 = (r11 / 255 * 0.98) + 0.1

    im22 = Image.open(os.getcwd()+"\\img\\222.JPG")
    r22 = np.dot(np.array(im22),[0.299, 0.587, 0.114]).ravel()
    scaled22 = (r22 / 255 * 0.98) + 0.1

    im33 = Image.open(os.getcwd()+"\\img\\333.JPG")
    r33 = np.dot(np.array(im33),[0.299, 0.587, 0.114]).ravel()
    scaled33 = (r33 / 255 * 0.98) + 0.1

    im44 = Image.open(os.getcwd()+"\\img\\444.JPG")
    r44 = np.dot(np.array(im44),[0.299, 0.587, 0.114]).ravel()
    scaled44 = (r44 / 255 * 0.98) + 0.1

    for i in range(40):
        # n.train(scaled0, [0.99,0.01,0.01,0.01,0.01])
        n.train(scaled1, [0.01,0.99,0.01,0.01,0.01])
        n.train(scaled2, [0.01,0.01,0.99,0.01,0.01])
        n.train(scaled3, [0.01,0.01,0.01,0.99,0.01])
        n.train(scaled4, [0.01,0.01,0.01,0.01,0.99])
        # n.train(scaled44, [0.01,0.01,0.01,0.01,0.99])
        print()

    # print(n.work(scaled0).ravel())
    print(n.work(scaled1).ravel())
    print(n.work(scaled2).ravel())
    print(n.work(scaled3).ravel())
    print(n.work(scaled4).ravel())



