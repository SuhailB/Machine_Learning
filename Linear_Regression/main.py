import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv('training_data.txt',names=["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","l"]) #read the data
#my_data = (my_data - my_data.mean())/my_data.std()

X = my_data.iloc[:,0:15]

ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = my_data.iloc[:,15:16].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,16])

#set hyper parameters
alpha = 0.01
iters = 2000
epsilon = 0.1
lambd = 1


#computecost
def computeCost(X,y,theta, lambd):
    sum1 = np.power(((X @ theta.T)-y),2)
    sum2 = np.power(theta,2)
    
    return np.sum(sum1)/(2 * len(X)) + lambd*np.sum(sum2)/(2*len(X))

#gradient descent
def gradientDescent(X,y,theta,alpha,epsilon,lambd):
    cost = np.zeros(2000)
    tempCost = 1000
    iter = 0
    while(tempCost>epsilon):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        
        cost[iter] = computeCost(X, y, theta, lambd)
        tempCost = (abs(cost[iter-1]-cost[iter])*100)/cost[iter-1]
        iter += 1
    
    return theta,cost, iter

#running the gd and cost function
g,cost, count = gradientDescent(X,y,theta,alpha,epsilon,lambd)
print(g)
print(count)


finalCost = computeCost(X,y,g,lambd)
print(finalCost)

def computeTestingCost(X,y,theta):
    sum1 = np.power(((X @ theta.T)-y),2)
    sum2 = np.power(theta,2)
    
    return np.sum(sum1)/(2 * len(X))

test_data = pd.read_csv('testing_data.txt',names=["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","l"]) #read the data
#my_data = (my_data - my_data.mean())/my_data.std()

X = test_data.iloc[:,0:15]

ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = test_data.iloc[:,15:16].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
theta = np.zeros([1,16])


print(computeTestingCost(X,y,g))

#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 