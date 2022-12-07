# -*- coding: UTF-8 -*-

import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from data_picture_util import plot_confusin_pictures ,read_pkl_data,read_ups_data
import numpy as np
from sklearn.metrics import accuracy_score


def main(X_train, y_train,X_test, y_test ,savename):

    cart = DecisionTreeClassifier()
    models = []
    model_log = LogisticRegression()
    models.append(('log',model_log))
    print("model1 created....")
    model_cart = DecisionTreeClassifier()
    models.append(('cart',model_cart))
    print("model2 created....")
    model_svc = SVC()
    models.append(('svm',model_svc))
    print("model3 created....")
    ensemble_model = VotingClassifier(estimators=models)
    ensemble_model.fit(X_train, y_train)
    accuracy=ensemble_model.score(X_train, y_train)
    print("model_traiing_accuracy %f"%accuracy)
    print("prediction....")
    predictions=ensemble_model.predict(X_test)
    print(predictions)
    predictions_accuracy=accuracy_score(y_test, predictions)
    print("model_testing_accuracy %f"%predictions_accuracy)
    plot_confusin_pictures(y_test,predictions,save_name='voting_confusion_matrix_%s.png'%savename)
if __name__ == '__main__':
    # # ********************************************************
    # _train, y_train, X_test, y_test = read_pkl_data()
    # main(X_train, y_train,X_test, y_test ,"data1")

    # # # ********************************************************
    X_train, y_train, _, _ = read_pkl_data()
    X_test, y_test ,_ ,_= read_ups_data()
    main(X_train, y_train ,X_test, y_test ,"data2")
    # ##################################################################################




