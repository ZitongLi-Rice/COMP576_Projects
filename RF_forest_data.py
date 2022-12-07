#-*- coding:utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_picture_util import plot_confusin_pictures ,read_pkl_data,read_ups_data
import matplotlib.pyplot as plt


def main(x_train_data,y_train_data,x_test_data,y_test_data,savename):
    accuracy_list = []
    size_list = []
    for n_estimator in range(10,200,10):
        classifier=RandomForestClassifier(n_estimators=n_estimator)
        classifier.fit(x_train_data,y_train_data)
        predictions = classifier.predict(x_test_data)
        plot_confusin_pictures(y_test_data,predictions,save_name='./RF/forest_confusion_matrix_estimator_%d_%s.png'%(n_estimator,savename))
        acc_rf = accuracy_score(y_test_data, predictions)
        print("n_estimators = %d, random forest accuracy:%f" % (n_estimator, acc_rf))
        accuracy_list.append(acc_rf * 100)
        size_list.append(n_estimator)
    plt.plot(size_list, accuracy_list, 'ro')
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.savefig("./RF/rf_accuracy.png")
    plt.show()
if __name__ == '__main__':
    # ***************************mnist training and mnist test****************************
    # x_train_data,y_train_data,x_test_data,y_test_data=read_pkl_data()
    # main(x_train_data, y_train_data, x_test_data, y_test_data, "data1")
    # **************************mnist training and usps test******************************
    x_train_data,y_train_data,_,_=read_pkl_data()
    x_test_data,y_test_data=read_ups_data()
    main(x_train_data, y_train_data, x_test_data, y_test_data,"data2")
