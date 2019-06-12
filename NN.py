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
W1 = (pd.read_csv('T_W1.csv', header=None)).values#(pd.read_csv('data\Initial_W1.csv', header=None)).values
W2 = (pd.read_csv('T_W2.csv', header=None)).values


#print(np.shape(W1))
#print(np.shape(W2))

def Logistic_Function(z):
    return 1 / (1 + np.exp(-z))

def Forward_Propagation(X,W1,W2):
    
    Z1 = X@W1.T
    H = Logistic_Function(Z1)
    
    ones = np.ones([H.shape[0],1])
    H = np.concatenate((ones,H),axis=1)
    
    Z2 = H@W2.T
    Y_ = Logistic_Function(Z2)

    return H,Y_,Z1

lambd = 3
def Loss_Function(X,Y,W1,W2,Y_):
    sum1 = np.sum(-1*Y*np.log(Y_)-(1-Y)*np.log(1-Y_))/len(X)
    sum2 = lambd*(np.sum(W1[:,1:]**2) + np.sum(W2[:,1:]**2))/(2*len(X))  
    return sum1+sum2
H,Y_,Z1 = Forward_Propagation(X,W1,W2)
print(Loss_Function(X,Y,W1,W2,Y_))

def Logistic_Gradient(z):
    return Logistic_Function(z)*(1-Logistic_Function(z))

###########################################################
#    back propagation


#print(H.shape)


def GW1Ji(X,Y,H,Y_,Z1,W2):
    B2 = (Y_ - Y)
    B1 = (B2@W2[:,1:])*Logistic_Gradient(Z1)
    GW1J = B1.T @ X
    return GW1J

def GW2Ji(X,Y,H,Y_,Z1):
    B2 = (Y_ - Y)
    GW2J = B2.T @ H
    return GW2J


def W1_Gradient(W1,H,Y_,Z1,W2):
    tempW = np.array(W1)
    term1 = (1/len(X))*GW1Ji(X,Y,H,Y_,Z1,W2)
    tempW[:,0] = 0
    term2 = (lambd/len(X))*W1
    return term1+term2

def W2_Gradient(W2,H,Y_,Z1):
    tempW = np.array(W2)
    term1 = (1/len(X))*GW2Ji(X,Y,H,Y_,Z1)
    tempW[:,0] = 0
    term2 = (lambd/len(X))*W2
    return term1+term2



def Gradient_Descent(X,Y,W1,W2):
    
    k=0
    cost = []
    while(k<10000):
        
        H,Y_,Z1 = Forward_Propagation(X,W1,W2)
        W1 = W1 - 0.2*W1_Gradient(W1,H,Y_,Z1,W2)
        W2 = W2 - 0.2*W2_Gradient(W2,H,Y_,Z1)
        cost.append(Loss_Function(X,Y,W1,W2,Y_)) 
        k += 1

    return W1,W2,cost,k

H,Y_,Z1 = Forward_Propagation(X,W1,W2)
print(Loss_Function(X,Y,W1,W2,Y_))

W1_,W2_,cost,count = Gradient_Descent(X,Y,W1,W2)

H,Y_,Z1 = Forward_Propagation(X,W1_,W2_)
print(Loss_Function(X,Y,W1_,W2_,Y_))


Y_Actual = np.array([np.where(r==1)[0][0] for r in Y]).reshape(5000,1)
Y_Predicted = np.array([np.where(r==max(r))[0][0] for r in Y_]).reshape(5000,1)

#print(Y_Actual)
#print(Y_Predicted)

Difference = Y_Actual - Y_Predicted

ones = np.ones([5000,1])
zeros = np.zeros([5000,1])

hits = np.where(Difference==0,ones,zeros)
#print(Difference)
print(np.sum(hits),5000,"%",100*(np.sum(hits)/5000))

np.savetxt("T_W1.csv", W1_, delimiter=",")
np.savetxt("T_W2.csv", W2_, delimiter=",")

fig, ax = plt.subplots()  
ax.plot(np.arange(count), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Loss function over iterations') 

#print(W1_test)
#for i in range (len(X)):
#    Z1[0]
#
#print(np.shape(Z1))
#def Forward_Propagation(x)
#
#from numpy import argmax
## define input string
#data = 'hello world'
#print(data)
## define universe of possible input values
#alphabet = 'abcdefghijklmnopqrstuvwxyz '
## define a mapping of chars to integers
#char_to_int = dict((c, i) for i, c in enumerate(alphabet))
#int_to_char = dict((i, c) for i, c in enumerate(alphabet))
## integer encode input data
#integer_encoded = [char_to_int[char] for char in data]
#print(integer_encoded)
## one hot encode
#onehot_encoded = list()
#for value in integer_encoded:
#	letter = [0 for _ in range(len(alphabet))]
#	letter[value] = 1
#	onehot_encoded.append(letter)
#print(onehot_encoded)
## invert encoding
#inverted = int_to_char[argmax(onehot_encoded[0])]
#print(inverted)