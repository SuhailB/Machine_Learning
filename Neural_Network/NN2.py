import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#prepare Y
raw_Y = pd.read_csv('data\Y.csv', header=None)
raw_Y = raw_Y.values
Y =  np.zeros((5000,10))
for i in range(5000):
	Y[i,(raw_Y[i]-1)]=1

#prepare X
X = pd.read_csv('data\X.csv', header=None)
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
#Y = pd.DataFrame.from_records(Y)
W1 = (pd.read_csv('data\initial_W1.csv', header=None)).values
W2 = (pd.read_csv('data\initial_W2.csv', header=None)).values

#print(np.shape(W1))
#print(np.shape(W2))

def Logistic_Function(z):
    return 1 / (1 + np.exp(-z))

Z1 = X@W1.T

print(X.shape,W1.T.shape)
H = Logistic_Function(Z1)

ones = np.ones([H.shape[0],1])
H = np.concatenate((ones,H),axis=1)

Z2 = H@W2.T
Y_ = Logistic_Function(Z2)
#print(Y[4999])
#print(Y_[4999])
#print(W1[:,:].shape)
#print(W1[:,1:])
lambd = 3
def Loss_Function(X,Y,W1,W2):
    sum1 = np.sum(-1*Y*np.log(Y_)-(1-Y)*np.log(1-Y_))/len(X)
    sum2 = lambd*(np.sum(W1[:,1:]**2) + np.sum(W2[:,1:]**2))/(2*len(X))  
    return sum1+sum2
print(Loss_Function(X,Y,W1,W2))

def Logistic_Gradient(z):
    return Logistic_Function(z)*(1-Logistic_Function(z))

###########################################################
#    back propagation


#print(H.shape)


def Gradients(i):
    B2 = (Y_[i,:]-Y[i,:]).reshape(1,10)
    B1 = (B2@W2[:,1:])*Logistic_Gradient(Z1[i,:])
    GW2J = B2.T@H[i,:].reshape(1,26)
    GW1J = B1.T@X[i,:].reshape(1,401)
    return GW1J, GW2J
G1, G2 = Gradients(0)
#print(G1.shape, G2.shape)
#print(W1[:,1:].shape, W2[:,1:].shape)

def W_Gradient(W,l):
    tempW = np.array(W)
    term1 = np.zeros(Gradients(0)[l-1].shape)
    for i in range(len(X)):
        term1 += Gradients(i)[l-1]
    term1 = (1/len(X))*term1
    print(term1[5,5])
    W[:,0] = 0
    term2 = (lambd/len(X))*W[:,:]
    return term1+term2


#print(W_Gradient(W1,1)[5,5])
print(W_Gradient(W2,2)[5,5])

#def Gradient_Descent(X,Y,W1,W2):
#    
#    k=0
#    
#    while(k<10):
#        W1 = W1 - 0.2*W_Gradient(W1,1)
#        W2 = W2 - 0.2*W_Gradient(W2,2)
#
#        k += 1
#
#    return W1,W2
#
#W1_,W2_ = Gradient_Descent(X,Y,W1,W2)
#print(W1_[0][0])
#print(Loss_Function(X,Y,W1_,W2_))
