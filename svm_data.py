#-*- coding:utf-8 -*-

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_picture_util import plot_confusin_pictures ,read_pkl_data,read_ups_data
import matplotlib.pyplot as plt



def main(x_train_data,y_train_data,x_test_data,y_test_data,savename):
    accuracy_list = []
    size_list = []
    i=0
    for gama in [0.1,0.15,0.2]:
        i+=1
        classifier=SVC(kernel='rbf',C=10.0,gamma=gama)
        classifier.fit(x_train_data,y_train_data)
        predictions = classifier.predict(x_test_data)
        plot_confusin_pictures(y_test_data,predictions,save_name='./SVM/SVM_confusion_matrix_gama_%d_%s.png'%(gama,savename))
        acc_rf = accuracy_score(y_test_data, predictions)
        print("gamma = %d, SVM  accuracy:%f" % (gama, acc_rf))
        accuracy_list.append(acc_rf * 100)
        size_list.append(i)

    plt.plot(size_list, accuracy_list, 'ro')
    plt.xlabel("gamma")
    plt.ylabel("Accuracy")
    plt.savefig("./SVM/SVM_accuracy.png")
    plt.show()
if __name__ == '__main__':
    # ***************************mnist training and mnist test****************************
    # x_train_data,y_train_data,x_test_data,y_test_data=read_pkl_data()
    # main(x_train_data, y_train_data, x_test_data, y_test_data, "data1")
    # **************************mnist training and usps test******************************
    x_train_data, y_train_data, _, _ = read_pkl_data()
    x_test_data, y_test_data = read_ups_data()
    main(x_train_data, y_train_data, x_test_data, y_test_data,"data2")