
#imports
from __future__ import division
import os, sys
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.metrics import *
from keras.models import *
from keras.layers import Dense, Activation
from keras.optimizers import *
import threading
import numpy as np

class LogModel(threading.Thread):
    """Threaded Logistic Regression Model"""
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        for i in range(6):
            X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
        for i in range(6):
            XT[:, i] = (XT[:, i] - XT[:, i].mean()) / (XT[:, i].std())
        logModel = LogisticRegression(C=10000)
        logModel.fit(X, Y)
        sd = logModel.predict(XT)
        recall = float(recall_score(YT, sd, average='macro'))
        precision = float(precision_score(YT, sd, average='macro'))
        F1_Score = 2*((precision*recall)/(precision+recall))
        print("Recall of Logistic Regression Model: %.2f" % (recall * 100) + " %")
        print("Precision of Logistic Regression Model: %.2f" % (precision *100) + " %")
        print("F1-Score of Logistic Regression Model: %.2f" % (F1_Score *100) + " %")
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of Logistic Regression Model: %.2f" % acc+' %')
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of Logistic Regression Model: %.2f" % (acc)+' %' + " \n" + "Precision of Logistic Regression Model: %.2f" % (precision * 100) + " %" + "\n" + "Recall of Logistic Regression Model: %.2f" % (recall *100) + " %" + "\n" + "F1-Score of Logistic Regression Model: %.2f" % (F1_Score *100) + " %")



class SVMModel(threading.Thread):
    """Threaded Support Vector Machine Model"""
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        for i in range(6):
            X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
        for i in range(6):
            XT[:, i] = (XT[:, i] - XT[:, i].mean()) / (XT[:, i].std())
        print("sv model basladi")
        svModel = SVC(kernel='rbf')
        svModel.fit(X, Y)
        sd = svModel.predict(XT)
        recall = float(recall_score(YT, sd, average='macro'))
        precision = float(precision_score(YT, sd, average='macro'))
        F1_Score = 2*((precision*recall)/(precision+recall))
        print("Recall of of SVM Model: %.2f" % (recall * 100) + " %")
        print("Precision of SVM Model: %.2f" % (precision *100) + " %")
        print("F1-Score of SVM Model: %.2f" % (F1_Score *100) + " %")
        acc =  (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of SVM Model: %.2f"%acc+' %')
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of SVM Model: %.2f" % (acc)+' %' + " \n" + "Precision of SVM Model: %.2f" % (precision * 100) + " %" + "\n" + "Recall of SVM Model: %.2f" % (recall *100) + " %" + "\n" + "F1-Score of SVM Model: %.2f" % (F1_Score *100) + " %")


class DTModel(threading.Thread):
    """Threaded Decision Tree Model"""
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        dtModel = DecisionTreeClassifier()
        dtModel.fit(X, Y)
        sd = dtModel.predict(XT)
        recall = float(recall_score(YT, sd, average='macro'))
        precision = float(precision_score(YT, sd, average='macro'))
        F1_Score = 2*((precision*recall)/(precision+recall))
        print("Recall of Decision Tree Model: %.2f" % (recall * 100) + " %")
        print("Precision of Decision Tree Model: %.2f" % (precision *100) + " %")
        print("F1-Score of Decision Tree Model: %.2f" % (F1_Score *100) + " %")
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of Decision Tree Model: %.2f" % acc+' %')
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of Decision Tree Model: %.2f" % (acc)+' %' + " \n" + "Precision of Decision Tree Model: %.2f" % (precision * 100) + " %" + "\n" + "Recall of Decision Tree Model: %.2f" % (recall *100) + " %")


class NBModel(threading.Thread):
    """Threaded Gaussian Naive Bayes Model"""
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        nbModel = GaussianNB()
        nbModel.fit(X, Y)
        sd = nbModel.predict(XT)
        recall = float(recall_score(YT, sd, average='macro'))
        precision = float(precision_score(YT, sd, average='macro'))
        F1_Score = 2*((precision*recall)/(precision+recall))
        print("Recall of Gaussian Naive Bayes Model: %.2f" % (recall * 100) + " %")
        print("Precision of Gaussian Naive Bayes Model: %.2f" % (precision *100) + " %")
        print("F1-Score of Gaussian Naive Bayes Model: %.2f" % (F1_Score *100) + " %")
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of Gaussian Naive Bayes Model: %.2f" % acc +' %')
        print('='*100)
        if self.accLabel: self.accLabel.set("Accuracy of Gaussian Naive Bayes Model: %.2f" % (acc)+' %' + " \n" + "Precision of Gaussian Naive Bayes Model: %.2f" % (precision * 100) + " %" + "\n" + "Recall of Gaussian Naive Bayes Model: %.2f" % (recall *100) + " %" + "\n" + "F1-Score of Gaussian Naive Bayes Model: %.2f" % (F1_Score *100) + " %")


class KNNModel(threading.Thread):
    """Threaded K Nearest Neighbours Model"""
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        for i in range(6):
            X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
        for i in range(6):
            XT[:, i] = (XT[:, i] - XT[:, i].mean()) / (XT[:, i].std())
        print("knn model basladi")
        knnModel = KNeighborsClassifier()
        knnModel.fit(X, Y)
        sd = knnModel.predict(XT)
        recall = float(recall_score(YT, sd, average='macro'))
        precision = float(precision_score(YT, sd, average='macro'))
        F1_Score = 2*((precision*recall)/(precision+recall))
        print("Recall of KNN Model: %.2f" % (recall * 100) + " %")
        print("Precision of KNN Model: %.2f" % (precision *100) + " %")
        print("F1-Score of KNN Model: %.2f" % (F1_Score *100) + " %")
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of KNN Model: %.2f" % acc+' %')
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of KNN Model: %.2f" % (acc)+' %' + " \n" + "Precision of KNN Model: %.2f" % (precision * 100) + " %" + "\n" + "Recall of KNN Model: %.2f" % (recall *100) + " %" + "\n" + "F1-Score of KNN Model: %.2f" % (F1_Score *100) + " %")


class ANNModel(threading.Thread):
    """Threaded Neural Network Model"""
    def __init__(self, X, Y, XT, YT, accLabel=None):
        threading.Thread.__init__(self)
        self.X = X
        self.Y = Y
        self.XT=XT
        self.YT=YT
        self.accLabel= accLabel

    def run(self):
        X = np.zeros(self.X.shape)
        Y = np.zeros(self.Y.shape)
        XT = np.zeros(self.XT.shape)
        YT = np.zeros(self.YT.shape)
        np.copyto(X, self.X)
        np.copyto(Y, self.Y)
        np.copyto(XT, self.XT)
        np.copyto(YT, self.YT)
        # X = self.X
        # Y = self.Y
        # XT = self.XT
        # YT = self.YT
        for i in range(6):
            X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
        for i in range(6):
            XT[:, i] = (XT[:, i] - XT[:, i].mean()) / (XT[:, i].std())

        model = Sequential()
        model.add(Dense(10, input_dim=6, activation="sigmoid"))
        model.add(Dense(10, activation='sigmoid'))
        model.add(Dense(1))
        sgd = SGD(lr=0.01, decay=0.000001, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                  loss='mse')
        model.fit(X, Y, nb_epoch=200, batch_size=100)
        sd = model.predict(XT)
        sd = sd[:, 0]
        sdList = []
        for z in sd:
            if z>=0.5:
                sdList.append(1)
            else:
                sdList.append(0)
        sdList = np.array(sdList)
        acc = (sum(sdList == YT) / len(YT) * 100)
        recall = float(recall_score(YT, sd, average='macro'))
        precision = float(precision_score(YT, sd, average='macro'))
        F1_Score = 2*((precision*recall)/(precision+recall))
        print("Recall of Gaussian Naive Bayes Model: %.2f" % (recall * 100) + " %")
        print("Precision of Gaussian Naive Bayes Model: %.2f" % (precision *100) + " %")
        print("F1-Score of Gaussian Naive Bayes Model: %.2f" % (F1_Score *100) + " %")
        acc = (sum(sd == YT) / len(YT) * 100)
        print("Accuracy of Gaussian Naive Bayes Model: %.2f" % acc +' %')
        print('='*100)
        print("Accuracy of ANN Model: %.2f" % acc+" %")
        
        print('=' * 100)
        if self.accLabel: self.accLabel.set("Accuracy of KNN Model: %.2f" % (acc)+' %' + " \n" + "Precision of KNN Model: %.2f" % (precision * 100) + " %" + "\n" + "Recall of KNN Model: %.2f" % (recall *100) + " %" + "\n" + "F1-Score of KNN Model: %.2f" % (F1_Score *100) + " %")
