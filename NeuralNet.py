#-*- coding:utf-8 -*-
import numpy as np


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
        self.activation_function = lambda x: 1/(1+np.exp(-x))

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
        # print(output_errors.mean())
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

    input_nodes = 784
    hidden_nodes = 100
    output_nodes =  10
    learning_rate = 20

    n = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # for i in range(100):
    #     n.train([0.5,0.5,0.5],[0.8,0.9])
    # print(n.work([0.5,0.5,0.5]))

    # load the mnist training data CSV file into a list
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    # train the neural network
    # epochs is the number of times the training data set is used for training
    epochs = 5
    for e in range(epochs):
        # go through all records in the training data set
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01

            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
        pass
    pass
    # # load the mnist test data CSV file into a list
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    # test the neural network
    # scorecard for how well the network performs, initially empty
    scorecard = []
    # go through all the records in the test data set
    for record in test_data_list:
    # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.work(inputs)
        # the index of the highest value corresponds to the label
        label = np.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
        # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
        pass
    pass
    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() /   scorecard_array.size)

