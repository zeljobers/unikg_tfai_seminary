from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp,tanh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network


def initialize_network_custom(tab):
    network = list()
    for idx_layer in range(1,len(tab)):
        layer = []
        for idx_neuron in range(tab[idx_layer]):
            randomWeight = []
            for k in range(tab[idx_layer-1]+1):
                randomWeight.append(random())
            temp = {'weights':randomWeight}
            layer.append(temp)
        network.append(layer)
    return network


def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def transfer_sigmoid(x, derivate):
    if derivate == 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return x * (1.0 - x)

def transfer_tanh(x, derivate):
    if derivate == 0:
        return tanh(x)
    else:
        return 1.0 - tanh(x)**2

def forward_propagate(network, row, transfer):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation, 0)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def backward_propagate_error(network, expected, transfer):
	for idx_layer in reversed(range(len(network))):
		layer = network[idx_layer]
		errors = list()
		if idx_layer != len(network)-1:
			for idx_neuron_layer_N in range(len(layer)):
				error = 0.0 
				for neuron_layer_M in network[idx_layer + 1]:
					error += (neuron_layer_M['weights'][idx_neuron_layer_N] * neuron_layer_M['delta'])
				errors.append(error)
		else:
			for idx_neuron in range(len(layer)):
				neuron = layer[idx_neuron]
				errors.append(expected[idx_neuron] - neuron['output'])
		for idx_neuron in range(len(layer)):
			neuron = layer[idx_neuron]
			neuron['delta'] = errors[idx_neuron] * transfer(neuron['output'], 1)

def update_weights(network, row, l_rate):
	for idx_layer in range(len(network)):
		inputs = row[:-1]
		if idx_layer != 0:
			inputs = [neuron['output'] for neuron in network[idx_layer - 1]]
		for neuron in network[idx_layer]:
			for idx_input in range(len(inputs)):
				neuron['weights'][idx_input] += l_rate * neuron['delta'] * inputs[idx_input]
			neuron['weights'][-1] += l_rate * neuron['delta'] * 1

def one_hot_encoding(n_outputs, row_in_dataset):
    expected = [0 for i in range(n_outputs)]
    expected[row_in_dataset[-1]] = 1
    return expected

def train_network(network, train, test, l_rate, n_epoch, n_outputs, transfer):
	accuracy=[]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row, transfer)
			expected = one_hot_encoding(n_outputs, row)
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected, transfer)
			update_weights(network, row, l_rate)
			accuracy.append(get_prediction_accuracy(network, test, transfer))
		accuracies.append(accuracy)

def predict(network, row, transfer):
	outputs = forward_propagate(network, row, transfer)
	return outputs.index(max(outputs))

def get_prediction_accuracy(network, train, transfer):
    predictions = list()
    for row in train:
        prediction = predict(network, row, transfer)
        predictions.append(prediction)
    expected_out = [row[-1] for row in train]
    accuracy = accuracy_metric(expected_out, predictions)
    return accuracy

def back_propagation(train, test, l_rate, n_epoch, n_hidden, transfer):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network_custom([n_inputs, 5, n_outputs])
	layerPrint=[]
	for i in range(len(network)):
		layerPrint.append(len(network[i]))
	print('network created: %d layer(s):' % len(network), layerPrint)
	train_network(network, train, test, l_rate, n_epoch, n_outputs, transfer)
	predictions = list()
	print ("perform predictions on %d set of inputs:" % len(test))
	for row in test:
		prediction = predict(network, row, transfer)
		predictions.append(prediction)
	print("pred =", predictions)
	return(predictions)

def load_csv():
	dataset = load_iris(as_frame=True).frame.values.tolist()	
	return dataset

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column])
 
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)        
		train_set.remove(fold)         
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:               
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, fold, *args)
		actual = [row[-1] for row in fold]
		print(actual)
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
		print('- Training[%d] performed' % len(scores))
		print('---------------------------------------')
	return scores
seed(1)

accuracies = list()

dataset = load_csv()
for i in range(len(dataset[0])-1):
 	str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
n_folds = 3
l_rate = 0.3
n_epoch = 500
n_hidden = 5
print('---------------------------------------')
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden, transfer_sigmoid)
print('Scores (per fold): %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
	
