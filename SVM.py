import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


my_data = pd.read_csv('data.txt', sep='\t')#,names=["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","l"]) #read the data

my_data = (my_data - my_data.mean())/my_data.std()

#prepare X matrix
X = my_data.iloc[:,0:8].values
#ones = np.ones([X.shape[0],1])
#X = np.concatenate((ones,X),axis=1) #X = pd.DataFrame.from_records(X)
##prepare y matrix
y = my_data.iloc[:,8:9].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
##prepare w matrix
w = np.zeros([1,8])

b = 0

C = 10


#cost/loss function
def Cost(X,y,w,b,C):
    #regular format expression
    sum1 = X @ w.T + b
    sum1 = y * sum1
    sum1 = 1-sum1  
    sum1 = sum1.clip(min=0)
    sum2 = np.power(w,2)
    return (np.sum(sum2)/2) + C*np.sum(sum1)

print("Cost",Cost(X,y,w,b,10))

def hinge_loss(w,x,y):
    """ evaluates hinge loss and its gradient at w

    rows of x are data points
    y is a vector of labels
    """
    loss,grad = 0,0
    for (x_,y_) in zip(x,y):
        v = y_*(np.dot(w,x_)+b)
#        print(v)
        loss += max(0,1-v)
        grad += 0 if v > 1 else -y_*x_
    return loss
print("loss",hinge_loss(w,X,y))

def L_w(X,y,w,b,j):
    z = X @ w.T + b
    z = y*z
    v = np.zeros([760,1])
    Xj = X[:,j]
    Xj = Xj.reshape(len(X),1)
    z = np.where(z>=1,v,-1*y*Xj)
    return z
 
def L_b(y,w,b):
    z = X @ w.T + b
    z = y*z
    v = np.zeros([760,1])
    z = np.where(z>=1,v,-1*y)
    return z

def Batch_Gradient_w(X,y,w,b,j):
    return w[0,j]+C*np.sum(L_w(X,y,w,b,j))

def Batch_Gradient_b(y,w,b):
    return b + C*np.sum(L_b(y,w,b))


def Stocastic_Gradient_w(X,y,w,b,j,i):
    Lw = L_w(X,y,w,b,j)
    return w[0,j] + C*Lw[i,0]

def Stocastic_Gradient_b(y,w,b,i):
    Lb = L_b(y,w,b)
    return b + C*Lb[i,0]



def Batch_SVM(X,y,w,b,C,alpha,epsilon):
    
    k=0
    cost = []
    tempCost = 10
    while(tempCost>epsilon):
        
        for j in range(len(w[0])):
          w[0][j] = w[0][j] - alpha*Batch_Gradient_w(X,y,w,b,j)
        b = b - alpha*Batch_Gradient_b(y,w,b)
        
        cost.append(Cost(X,y,w,b,C))
        
        if(k != 0):
            tempCost = (abs(cost[k-1]-cost[k])*100)/cost[k-1]
          
        k += 1
    return w,cost,k,b



def Stocastic_SVM(X,y,w,b,C,alpha,epsilon):
    
    k=0
    i=0
    cost = []
    dCost = []
    criteria = 10
    tempCost = 10
    m = len(X)
    while(k<1000):
        
        for j in range(len(w[0])):
          w[0][j] = w[0][j] - alpha*Stocastic_Gradient_w(X,y,w,b,j,i)
        b = b - alpha*Stocastic_Gradient_b(y,w,b,i)
        
        cost.append(Cost(X,y,w,b,C))
        
#        if(k>=1):
#            dCost.append(0.5*dCost[k-1]+0.5*tempCost)
        
        if(k != 0):
            tempCost = (abs(cost[k-1]-cost[k])*100)/cost[k-1]
#            criteria = dCost[k]
        
        
        
        i = (i+1)%m
        k += 1
    return w,cost,k,b
w = np.zeros([1,8])
b = 0
alpha = 0.00001
epsilon = 0.0004
w_,cost1,count1,b_ = Batch_SVM(X,y,w,b,C,alpha,epsilon)
print(Cost(X,y,w_,b_,C))
w = np.zeros([1,8])
b = 0
alpha = 0.001
epsilon = 0.0004
w_,cost2,count2,b_ = Stocastic_SVM(X,y,w,b,C,alpha,epsilon)
print(Cost(X,y,w_,b_,C))





#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(count1), cost1, 'r')
ax.plot(np.arange(count2), cost2, 'b')
    
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Loss function over iterations') 


#print(Gradient(X,y,w,b,0))
###
##print the loss function value before training
#initialCost = Cost(X,y,theta,lambd)
#print("Cost function before training",initialCost)
#
##gradient function Lasso Regularization (partial derivative with respect to thetaj)
#def Gradient(X,y,theta,lambd,j):
#  m = len(X)
#  Xj = X[:,j]
#  Xj = Xj.reshape(len(X),1)
#  sum = ((X @ theta.T)-y)*Xj
#  if(theta[0][j]==0):
#    lasso = 1
#  else:
#    lasso = (lambd*theta[0][j])/(2*m*abs(theta[0][j]))
#  
#  return (np.sum(sum)/m)+lasso
##linear Regression with Quadratic Regularization
#def LinearRegression(X,y,theta,alpha,epsilon,lambd):
#    cost = []
#    k = 0
#    tempCost = 10
#    while(tempCost>epsilon):
#        
#        for j in range(len(theta[0])):
#          theta[0][j] = theta[0][j] - alpha*Gradient(X,y,theta,lambd,j)
#        
#        cost.append(Cost(X, y, theta, lambd))
#        
#        if(k != 0):
#          tempCost = (abs(cost[k-1]-cost[k])*100)/cost[k-1]
#          
#        k += 1
#    
#    return theta,cost, k
#  
##running the gd and cost function
#g,cost, count = LinearRegression(X,y,theta,alpha,epsilon,lambd)
#
##print loss/cost function after training
#finalCost = Cost(X,y,g,lambd)
#print("Cost function after training",finalCost)
#
#
#def TestingCost(X,y,theta):
#  
#    sum1 = np.power(((X @ theta.T)-y),2)
#    
#    return np.sum(sum1)/(2 * len(X))
#
#test_data = pd.read_csv('raw_testing_data.txt',names=["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","l"]) #read the data
#test_data = (test_data - test_data.mean())/test_data.std()
#
#X = test_data.iloc[:,0:15]
#
#ones = np.ones([X.shape[0],1])
#X = np.concatenate((ones,X),axis=1)
#
#y = test_data.iloc[:,15:16].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
#theta = np.zeros([1,16])
#
#
#print("Cost function of testing data",TestingCost(X,y,g))
#
#def Rounding(theta):
#  for i in range(len(theta[0])):
#      if(abs(g[0][i])<0.005): theta[0][i] = 0
#
##print trained parameters of theta
#print(g[0])
#Rounding(g)
#print(g[0])
##plot the cost
#fig, ax = plt.subplots()  
#ax.plot(np.arange(count), cost, 'r')  
#ax.set_xlabel('Iterations')  
#ax.set_ylabel('Cost')  
#ax.set_title('Loss function over iterations') 