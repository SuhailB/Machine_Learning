#from mnist import MNIST
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def getData(train_ratio, test_size, normalize=True):
    
    data = pd.read_csv('train.csv').values

    TRAINING_SIZE = (int)(len(data)*train_ratio)
    
    X = data[:,1:]
    Y = data[:,0]
    
    print(X.max(),TRAINING_SIZE,test_size)
    
    if(normalize==True): X = X/X.max()
    
    
    X_train = X[0:TRAINING_SIZE,:]
    Y_train = Y[0:TRAINING_SIZE]
    
    X_test = X[TRAINING_SIZE+1:TRAINING_SIZE+test_size,:]
    Y_test = Y[TRAINING_SIZE+1:TRAINING_SIZE+test_size]
    
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = getData(0.8, 1001, True)

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, Y_train)

print("Compute predictions")
predicted = clf.predict(X_test)

print("Accuracy: ", accuracy_score(Y_test, predicted))

print("score of testing data ", clf.score(X_test, Y_test))



print(Y_test[0:10])
print(clf.predict(X_test[0:10,:]))



