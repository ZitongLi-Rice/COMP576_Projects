#-*- coding:utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from data_picture_util import plot_confusin_pictures ,read_pkl_data,read_ups_data

# tile() copies 9 rows of the matrix in horizontal direction, with the vertical direction unchanged
# zeros() generates a zero matrix with 784 elements
alpha = 0.001
weights = np.tile(np.zeros(784), (9, 1))


def post_prob(x, y): # Calculate the accuracy of the label
    if y == 9:
        return 1 / sum_exps(x)
    else:
        return np.exp(np.dot(weights[y], x)) / sum_exps(x)

def sum_exps(x):
    exp_array = []
    for weight in weights:
        exp_array.append(np.exp(np.dot(weight, x)))
    return (1 + np.sum(exp_array))

def gradient_ascent(p, y, x):
    for i in range(len(weights)):
        if y == i:
            weights[i] = np.add(np.array(weights[i]), np.array(alpha * (1 - p[i]) * np.array(x)))
        else:
            weights[i] = np.add(np.array(weights[i]), np.array(alpha * (-1) * p[i] * np.array(x)))

def train(x_train_data,y_train_data):
    for i in range(len(x_train_data)):
        predictions = []
        for j in range(10):
            predictions.append(round(post_prob(x_train_data[i], j), 2))
        gradient_ascent(predictions, y_train_data[i], x_train_data[i])


def iteration_result(iters,x_test_data,y_test_data,savename):
    predictions = []
    for i in range(len(x_test_data)):
        p = []
        for j in range(10):
            p.append(round(post_prob(x_test_data[i], j), 2))
        index = np.argmax(p)
        predictions.append(index)
    count = 0
    plot_confusin_pictures(y_test_data,predictions,save_name='E:/LR/logiression_confusion_matrix_iteration_%d_%s.png'%(iters,savename))
    for i in range(len(predictions)):
        if predictions[i] == y_test_data[i]:
            count += 1
    return (count / 10000)

def main(x_train_data,y_train_data,x_test_data,y_test_data,savename):
    accuracy_list = []
    size_list = []
    for i in range(50):
        train(x_train_data,y_train_data)
        accuracy = iteration_result(i,x_test_data,y_test_data,savename)
        print("For iteration = ", i + 1, " accuracy = ", accuracy * 100)
        accuracy_list.append(accuracy * 100)
        size_list.append((i + 1))

    plt.plot(size_list, accuracy_list, 'ro')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.savefig("./LR/logistic_accuracy_%s.png"%savename)
    plt.show()

if __name__ == "__main__":

    x_train_data,y_train_data,x_test_data,y_test_data=read_pkl_data()
    main(x_train_data, y_train_data, x_test_data, y_test_data,"data1")

    # x_train_data,y_train_data,_,_=read_pkl_data()
    # x_test_data,y_test_data=read_ups_data()
    # main(x_train_data, y_train_data, x_test_data, y_test_data,"data2")