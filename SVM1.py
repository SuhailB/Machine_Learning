import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

my_data = pd.read_csv('data1.txt', sep='\t')#,names=["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","l"]) #read the data
from sklearn.datasets.samples_generator import make_blobs

#my_data = (my_data - my_data.mean())/my_data.std()

#prepare X matrix
X = my_data.iloc[:,0:2].values
#ones = np.ones([X.shape[0],1])
#X = np.concatenate((ones,X),axis=1) #X = pd.DataFrame.from_records(X)
##prepare y matrix
y = my_data.iloc[:,2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
##prepare w matrix
w = np.zeros([1,2])

b = 1

C = 10
alpha = 0.01
epsilon = 0.04

(X,y) =  make_blobs(n_samples=50,n_features=2,centers=2,cluster_std=1.05,random_state=40)
y = np.where(y==0,-1,1)
y = y.reshape(50,1)

#plt.scatter(X[:,0],X[:,1], marker='o',c=y[:,0])
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
    v = np.zeros([50,1])
    Xj = X[:,j]
    Xj = Xj.reshape(len(X),1)
    z = np.where(z>=1,v,-1*y*Xj)
    return z
 
def L_b(y,w,b):
    z = X @ w.T + b
    z = y*z
    v = np.zeros([50,1])
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
    return w,b,cost,k

def Stocastic_SVM(X,y,w,b,C,alpha,epsilon):
    
    k=0
    i=0
    cost = []
    tempCost = 10
    m = len(X)
    while(k<10000000):
        
        for j in range(len(w[0])):
          w[0][j] = w[0][j] - alpha*Stocastic_Gradient_w(X,y,w,b,j,i)
        b = b - alpha*Stocastic_Gradient_b(y,w,b,i)
        
        cost.append(Cost(X,y,w,b,C))
        
        if(k != 0):
            tempCost = (abs(cost[k-1]-cost[k])*100)/cost[k-1]
        i = (i+1)%m
        k += 1
    return w,b,cost,k

w_,b_,c,count = Batch_SVM(X,y,w,b,C,alpha,epsilon)
print(Cost(X,y,w_,b_,C))


w = w_[0]
b = b_
postiveX=[]
negativeX=[]
for i,v in enumerate(y):
    if v==-1:
        negativeX.append(X[i])
    else:
        postiveX.append(X[i])

data_dict = {-1:np.array(negativeX), 1:np.array(postiveX)} 

max_feature_value=float('-inf')
min_feature_value=float('+inf')
        
for yi in data_dict:
    if np.amax(data_dict[yi])>max_feature_value:
        max_feature_value=np.amax(data_dict[yi])
                
    if np.amin(data_dict[yi])<min_feature_value:
        min_feature_value=np.amin(data_dict[yi])

colors = {1:'r',-1:'b'}
fig = plt.figure()
ax = fig.add_subplot(1,1,1)



def visualize(data_dict):
       
        print(w,b)
        #[[ax.scatter(x[0],x[1],s=100,color=colors[i]) for x in data_dict[i]] for i in data_dict]
        
        plt.scatter(X[:,0],X[:,1],marker='o',c=y[:,0])

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane_value(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (min_feature_value*0.9,max_feature_value*1.)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane_value(hyp_x_min, w, b, 1)
        psv2 = hyperplane_value(hyp_x_max, w, b, 1)
        ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane_value(hyp_x_min, w, b, -1)
        nsv2 = hyperplane_value(hyp_x_max, w, b, -1)
        ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane_value(hyp_x_min, w, b, 0)
        db2 = hyperplane_value(hyp_x_max, w, b, 0)
        ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')
        
        plt.axis([-5,10,-12,-1])
        plt.show()
     
visualize(data_dict)
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