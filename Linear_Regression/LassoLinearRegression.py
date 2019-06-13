import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv('raw_training_data.txt',names=["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","l"]) #read the data
my_data = (my_data - my_data.mean())/my_data.std()

#prepare X matrix
X = my_data.iloc[:,0:15]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

#prepare y matrix
y = my_data.iloc[:,15:16].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray

#prepare theta matrix
theta = np.zeros([1,16])

#set parameters
alpha = 0.01
epsilon = 0.1
lambd = 1

#cost/loss function
def Cost(X,y,theta, lambd):
  #regular format expression
  m = len(X)
  #sum1 = np.power(((X @ theta.T)-y),2)
  #sum2 = abs(theta)

#############################################################################
  #matrix expression
  J = ((y.T-theta@X.T)@(y.T-theta@X.T).T)/(2*m)
  sum2 = abs(theta)
  #return for regular format
  #return np.sum(sum1)/(2*len(X))+lambd*np.sum(sum2)/(2*len(X))
  #return for matrix format
  return  np.asscalar(J+lambd*np.sum(sum2)/(2*len(X)))

#print the loss function value before training
initialCost = Cost(X,y,theta,lambd)
print("Cost function before training",initialCost)

#gradient function Lasso Regularization (partial derivative with respect to thetaj)
def Gradient(X,y,theta,lambd,j):
  m = len(X)
  Xj = X[:,j]
  Xj = Xj.reshape(len(X),1)
  sum = ((X @ theta.T)-y)*Xj
  if(theta[0][j]==0):
    lasso = 1
  else:
    lasso = (lambd*theta[0][j])/(2*m*abs(theta[0][j]))
  
  return (np.sum(sum)/m)+lasso
#linear Regression with Quadratic Regularization
def LinearRegression(X,y,theta,alpha,epsilon,lambd):
    cost = []
    k = 0
    tempCost = 10
    while(tempCost>epsilon):
        
        for j in range(len(theta[0])):
          theta[0][j] = theta[0][j] - alpha*Gradient(X,y,theta,lambd,j)
        
        cost.append(Cost(X, y, theta, lambd))
        
        if(k != 0):
          tempCost = (abs(cost[k-1]-cost[k])*100)/cost[k-1]
          
        k += 1
    
    return theta,cost, k
  
#running the gd and cost function
g,cost, count = LinearRegression(X,y,theta,alpha,epsilon,lambd)

#print loss/cost function after training
finalCost = Cost(X,y,g,lambd)
print("Cost function after training",finalCost)


def TestingCost(X,y,theta):
  
    sum1 = np.power(((X @ theta.T)-y),2)
    
    return np.sum(sum1)/(2 * len(X))

test_data = pd.read_csv('raw_testing_data.txt',names=["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","l"]) #read the data
test_data = (test_data - test_data.mean())/test_data.std()

X = test_data.iloc[:,0:15]

ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = test_data.iloc[:,15:16].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,16])


print("Cost function of testing data",TestingCost(X,y,g))

def Rounding(theta):
  for i in range(len(theta[0])):
      if(abs(g[0][i])<0.005): theta[0][i] = 0

#print trained parameters of theta
print(g[0])
Rounding(g)
print(g[0])
#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(count), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Loss function over iterations') 