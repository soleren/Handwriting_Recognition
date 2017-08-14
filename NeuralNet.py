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

    # input_nodes = 400
    # hidden_nodes = 100
    # output_nodes =  5
    # learning_rate = 0.4
    input_nodes = 40000
    hidden_nodes = 300
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

    for i in range(40):
        # n.train(scaled0, [0.99,0.01,0.01,0.01,0.01])
        n.train(scaled1, [0.01,0.99,0.01,0.01,0.01])
        n.train(scaled2, [0.01,0.01,0.99,0.01,0.01])
        n.train(scaled3, [0.01,0.01,0.01,0.99,0.01])
        n.train(scaled4, [0.01,0.01,0.01,0.01,0.99])
        print()

    # print(n.work(scaled0).ravel())
    print(n.work(scaled1).ravel())
    print(n.work(scaled2).ravel())
    print(n.work(scaled3).ravel())
    print(n.work(scaled4).ravel())


    # for i in range(100):
    #     n.train([0.2,0.2,0.2], [0.2,0.2,0.2])
    #     n.train([0.9,0.9,0.9], [0.9,0.9,0.9])
    # print(n.work([0.3,0.3,0.3]))
    # print(n.work([0.7,0.7,0.7]))
    # print(n.work([0.1,0.1,0.1]))


    # # load the mnist training data CSV file into a list
    # training_data_file = open("mnist_train.csv", 'r')
    # training_data_list = training_data_file.readlines()
    # training_data_file.close()
    # # train the neural network
    # # epochs is the number of times the training data set is used for training
    # epochs = 5
    # for e in range(epochs):
    #     # go through all records in the training data set
    #     for record in training_data_list:
    #         # split the record by the ',' commas
    #         all_values = record.split(',')
    #         # scale and shift the inputs
    #         inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #         # create the target output values (all 0.01, except the desired label which is 0.99)
    #         targets = np.zeros(output_nodes) + 0.01
    #
    #         # all_values[0] is the target label for this record
    #         targets[int(all_values[0])] = 0.99
    #         n.train(inputs, targets)
    #     pass
    # pass
    # # # load the mnist test data CSV file into a list
    # test_data_file = open("mnist_test.csv", 'r')
    # test_data_list = test_data_file.readlines()
    # test_data_file.close()
    # # test the neural network
    # # scorecard for how well the network performs, initially empty
    # scorecard = []
    # # go through all the records in the test data set
    # for record in test_data_list:
    # # split the record by the ',' commas
    #     all_values = record.split(',')
    #     # correct answer is first value
    #     correct_label = int(all_values[0])
    #     # scale and shift the inputs
    #     inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #     # query the network
    #     outputs = n.work(inputs)
    #     # the index of the highest value corresponds to the label
    #     label = np.argmax(outputs)
    #     # append correct or incorrect to list
    #     if (label == correct_label):
    #     # network's answer matches correct answer, add 1 to scorecard
    #         scorecard.append(1)
    #     else:
    #     # network's answer doesn't match correct answer, add 0 to scorecard
    #         scorecard.append(0)
    #     pass
    # pass
    # # calculate the performance score, the fraction of correct answers
    # scorecard_array = np.asarray(scorecard)
    # print ("performance = ", scorecard_array.sum() /   scorecard_array.size)

