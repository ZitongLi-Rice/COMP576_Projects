#-*- coding:utf-8 -*-

import keras
from keras.models import Sequential                                                         
from keras.layers import  Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from data_picture_util import plot_confusin_pictures ,read_pkl_data,read_ups_data
import numpy as np
import matplotlib.pyplot as plt

def trans_data(data_y):
    result=[]
    for data in data_y:
        for i in np.arange(10):
            if data[i]==1:
                result.append(i)
                continue
    return result


def main(X_train, y_train, X_test, y_test,savename):
    tBatchSize = 128
    '''Step 1: Choose a model'''
    model = Sequential()  # Adopt a sequential model
    '''Step 2: Build the network layer'''
    '''Building a network just builds a network structure and defines the parameters of the network. There is no data set yet.'''
    # The input layer
    model.add(Dense(500, input_shape=(784,)))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))# Training was conducted with 50% of the rinks selected at random

    # Hidden layer, 500 nodes per each hidden layer
    model.add(Dense(500))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(500))
    model.add(Activation('tanh'))
    # Output layer
    model.add(Dense(10))  # 10 class
    model.add(Activation('softmax'))
    '''Step 3: network optimization and compilation'''
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    '''Step 4: training'''
    X_train = X_train.reshape(len(X_train), 784)
    X_test = X_test.reshape(len(X_test), 784)
    Y_train = (np.arange(10) == y_train[:, None]).astype(int)
    Y_test = (np.arange(10) == y_test[:, None]).astype(int)
    history=model.fit(X_train, Y_train, batch_size=tBatchSize, epochs=5, shuffle=True, verbose=2)
    model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
    accuracy_list_train = history.history["accuracy"]
    loss_list = history.history["loss"]
    nums_list = range(len(accuracy_list_train))
    plt.ylim(-0.1, 1.1)
    plt.plot(nums_list, accuracy_list_train, color="green", label='accuracy')
    plt.plot(nums_list, loss_list, color='blue', label='loss')
    plt.xlabel('epoches')
    plt.ylabel('accuracy and loss')
    plt.legend(loc="center right")
    plt.savefig("accuracy.png")
    plt.show()
    '''Step 5: output'''
    print("test set")
    scores = model.evaluate(X_test, Y_test, batch_size=tBatchSize, verbose=0)
    print("")
    print("The test accuracy is %f" , scores[1])
    predictions = model.predict(X_test, batch_size=tBatchSize, verbose=1)
    predictions= np.argmax(predictions, axis = 1)
    Y_test=np.argmax(Y_test, axis = 1)
    plot_confusin_pictures(Y_test,predictions,save_name='./KERAS/KERAS_confusion_matrix_%s.png'%savename)

if __name__ == '__main__':
    #***************************mnist training and mnist test****************************
    # X_train, y_train, X_test, y_test =read_pkl_data()
    # main(X_train, y_train, X_test, y_test, "data1")
    #**************************mnist training and usps test******************************
    X_train,y_train=read_ups_data()
    X_test, y_test=read_ups_data()
    main(X_train, y_train, X_test, y_test, "data2")
