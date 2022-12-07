#-*- coding:utf-8 -*-

from __future__ import division
import pickle
import gzip
from PIL import Image
import os
import _pickle as cPickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']
tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusin_pictures(y_true,y_pred,save_name='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print cm_normalized
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig(save_name, format='png')
    plt.show()

def read_pkl_data():
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')
    f.close()
    y_train_data = training_data[1]
    x_train_data = training_data[0]
    y_test_data = test_data[1]
    x_test_data = test_data[0]
    return x_train_data,y_train_data,x_test_data,y_test_data

USPSMat  = []
USPSTar  = []
curPath  = 'E:/USPSdata/Numerals'
savedImg = []
i=0
for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        i+=1
        print ("dealing %d ........"%i)
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)


write_file=open('ups_data.pkl','wb')
cPickle.dump(USPSMat,write_file,-1)
cPickle.dump(USPSTar,write_file,-1)
write_file.close()

USPSMat1  = []
USPSTar1  = []
curPath1  = 'E:/USPSdata/Test'
savedImg1 = []
i=0
for j in range(0,10):
    curFolderPath = curPath1 + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        i+=1
        print ("dealing %d ........"%i)
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg1 = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat1.append(imgdata)
            USPSTar1.append(j)


write_file=open('ups_data.pkl1','wb')
cPickle.dump(USPSMat1,write_file,-1)
cPickle.dump(USPSTar1,write_file,-1)
write_file.close()


def read_ups_data():
    read_file=open('ups_data.pkl','rb')
    data_x=cPickle.load(read_file)

def read_ups_data():
    read_file = open('ups_data.pkl1', 'rb')
    data_x = cPickle.load(read_file)
    data_y=cPickle.load(read_file)
    return data_x,data_y

if __name__ == '__main__':
    read_pkl_data()
    read_ups_data()