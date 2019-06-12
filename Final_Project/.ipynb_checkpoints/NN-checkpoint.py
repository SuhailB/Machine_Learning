# Plot ad hoc mnist instances
#from keras.datasets import mnist
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def getData(train_ratio, test_size, test_all=False, normalize=True):
    
    data = pd.read_csv('train.csv').values

    TRAINING_SIZE = (int)(len(data)*train_ratio)
    
    X = data[:,1:]
    Y = data[:,0]
    
    print(X.max(),TRAINING_SIZE,test_size)
    
    if(normalize==True): X = X/X.max()
    
    
    X_train = X[0:TRAINING_SIZE,:]
    Y_train = Y[0:TRAINING_SIZE]
    if test_all==True:
        X_test = X
        Y_test = Y
    else:
        X_test = X[TRAINING_SIZE+1:TRAINING_SIZE+test_size,:]
        Y_test = Y[TRAINING_SIZE+1:TRAINING_SIZE+test_size]
    
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = getData(0.8, 1001, False, True)


clf = MLPClassifier(solver='lbfgs', activation = 'logistic', alpha=1e-5,
                    hidden_layer_sizes=(16, 16), random_state=1, warm_start=True)

clf.fit(X_train, Y_train)                         

print("Compute predictions")
predicted = clf.predict(X_test)

print("Accuracy: ", accuracy_score(Y_test, predicted))

#print("score of training data ", clf.score(X, y))
print("score of testing data ", clf.score(X_test, Y_test))

#print(y[0:10])
#print(clf.predict(X[0:10,:]))


print(Y_test[0:10])
print(clf.predict(X_test[0:10]))


## Pick the fifth image from the dataset (it's a 9)
#i = 4
#image, label = X[i], y[i]
#
## Print the image
#output = Image.new("L", (28, 28))
#output.putdata(image)
#output.save("output.png")
#
## Print label
#print(label)

#
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#
#
#from sklearn.neural_network import MLPClassifier
#
##prepare Y
#y = pd.read_csv('data\Y.csv', header=None).values
##raw_Y = raw_Y.values
##Y =  np.zeros((5000,10))
##for i in range(5000):
##	Y[i,(raw_Y[i]-1)]=1
#
##prepare X
#X = pd.read_csv('data\X.csv', header=None).values
##ones = np.ones([X.shape[0],1])
##X = np.concatenate((ones,X),axis=1)
#
##data = pd.read_csv('train.csv').values
#
##X = data[:,1:]
##X = X/X.max()
##y = data[:,0]
#
##X = [[0., 0.], [1., 1.]]
##X = np.array(X)
##y = [0, 1]
##y = np.array(y)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                    hidden_layer_sizes=(25), random_state=1, warm_start = True)
#
#print(X.shape)
#print(y.shape)
#clf.fit(X, y)                         
#
#
#print(clf.score(X, y))
#
#print(y[0:10])
#print(clf.predict(X[0:10,:]))
#
#W1 = np.array(clf.coefs_[0]).reshape(25,400)
#W2 = np.array(clf.coefs_[1]).reshape(10,25)
#B1 = np.array(clf.intercepts_[0]).reshape(25,1)
#B2 = np.array(clf.intercepts_[1]).reshape(10,1)
#W1 = np.concatenate((B1,W1),axis=1)
#W2 = np.concatenate((B2,W2),axis=1)
##W3 = np.array(clf.coefs_[2])
#print(W1.shape)
#print(W2.shape)
#
#
#
##
##
###prepare Y
##raw_Y = pd.read_csv('data\Y.csv', header=None)
##raw_Y = raw_Y.values
##Y =  np.zeros((5000,10))
##for i in range(5000):
##	Y[i,(raw_Y[i]-1)]=1
##
###prepare X
##X = pd.read_csv('data\X.csv', header=None)
##ones = np.ones([X.shape[0],1])
##X = np.concatenate((ones,X),axis=1)
##
###initialize weights
#W1 = (pd.read_csv('data/initial_W1.csv', header=None)).values#(pd.read_csv('data\Initial_W1.csv', header=None)).values
#W2 = (pd.read_csv('data/initial_W2.csv', header=None)).values
#
#print(W1.shape)
#print(W2.shape)
#
#np.savetxt("Model_W1.csv", W1, delimiter=",")
#np.savetxt("Model_W2.csv", W2, delimiter=",")