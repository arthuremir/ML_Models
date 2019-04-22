
#%%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score


#%%
dataset = load_iris()
dir(dataset)


#%%
from numpy.random import permutation

shuffled_idx = permutation(range(dataset.data.shape[0]))


#%%
dataset.data = dataset.data[shuffled_idx]
dataset.target = dataset.target[shuffled_idx]


#%%
X = dataset.data
y = dataset.target


#%%
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(dataset.target.reshape(-1,1))


#%%
X = X/(X.max(axis=0))

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X,y,test_size=0.2)


#%%



#%%



#%%



#%%



#%%



#%%



#%%
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(3,),activation='relu')


#%%
mlp.fit(x_train,y_train)


#%%
mlp.predict(x_val[25].reshape(1,-1))


#%%
y_val[25]


#%%



#%%



#%%



#%%
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

class littleNN(BaseEstimator):
    def __init__(self, num_hidden, epochs=1000, num_batches=1, 
                 warm_start=False, plot_error_curve=False, verbose=0):
        self.num_hidden = num_hidden
        self.epochs = epochs
        self.num_batches = num_batches
        self.warm_start = warm_start
        self.plot_error_curve = plot_error_curve
        self.verbose = verbose
        self.num_weights_changed = 0
        self.error_curve = []
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def sigmoidPrime(self, x):
        return x*(1 - x)

    def correctWeights(self, hidden_layer, output_layer, y_true):

        delta2 = (output_layer - y_true) * self.sigmoidPrime(output_layer)
        self.w2 = self.w2 - hidden_layer.T.dot(delta2)
        
        delta1 = delta2.dot(self.w2[:-1].T)*self.sigmoidPrime(hidden_layer[:,:-1])
        self.w1 = self.w1 - self.input_layer.T.dot(delta1)
        
        self.num_weights_changed +=1
        
        
    def fit(self, X, y):

        m = X.shape[0]
        num_input = X.shape[1]
        num_output = y.shape[1]
       
        samples_per_batch = m // self.num_batches
        
        
        if (self.warm_start==False) or (self.warm_start==True and self.num_weights_changed==0):
            self.w1 = 2 * np.random.rand(num_input+1, self.num_hidden) - 1
            self.w2 = 2 * np.random.rand(self.num_hidden+1, num_output) - 1
        
        for i in range(self.epochs):
            #print('Epoch {0}'.format(i))
            
            for sample, y_true in zip(X,y): #zip(X[k:k+samples_per_batch,:], y[k:k+samples_per_batch,:]):
                
                self.input_layer = np.r_[sample,1].reshape(1,-1)
                z1 = np.c_[self.input_layer.dot(self.w1),1].reshape(1,-1)
                hidden_layer = self.sigmoid(z1)
                z2 = hidden_layer.dot(self.w2)
                output_layer = self.sigmoid(z2)
                
                E = np.sum((output_layer - y_true)**2)
                self.error_curve.append(E)

                self.correctWeights(hidden_layer, output_layer, y_true.reshape(1,-1))
                
                #k += samples_per_batch
        
        if self.plot_error_curve==True:
            plt.plot(np.arange(epochs*m), self.error_curve)

                
    def predict(self, X, proba=False, threshold=0.5):
        res = np.zeros((X.shape[0], 3))
        
        for i, sample in enumerate(X):
            self.input_layer = np.r_[sample,1].reshape(1, -1)
            hidden_layer = self.sigmoid((np.c_[self.input_layer.dot(self.w1),1].reshape(1, -1)))
            output_layer = self.sigmoid(hidden_layer.dot(self.w2))
            res[i] = output_layer
            if not proba:
                res[i] = np.where(res[i]>=threshold, 1, 0)
        
        return res


#%%
nn = littleNN(10)
nn.fit(x_train,y_train,plot_error_curve=True,epochs=1000)


#%%
len(nn.error_curve)


#%%
plt.scatter(np.arange(120), 
            np.array(nn.error_curve).reshape(1000,-1).mean(axis=0),
            s = 10
           )


#%%
plt.scatter(np.arange(100), 
            np.array(nn.error_curve).reshape(100,-1)[:,24],
            s = 10
           )


#%%
plt.scatter(np.arange(120), 
            np.array(nn.error_curve).reshape(1000,-1)[99,:],
            s = 10
           )


#%%
np.array(nn.error_curve).reshape(1000,-1)[99,:].argmax()


#%%
x_train[25]


#%%
res = nn.predict(x_val)

c = 0
for i in range(res.shape[0]):
    if(all(res[i]==y_val[i])):
        c+=1

c/res.shape[0]


#%%
nn = littleNN(10)
cross_val_score(nn,X,y,cv=3,scoring='accuracy')


#%%
Out[71].mean()


#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%
x = np.array([1,2,3,10])
np.where(x>2, 0,1)


#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
cross_val_score(lr,X,y,cv=5,scoring='accuracy')


#%%
nn = littleNN(3)
nn.fit(x_train,y_train)
res = nn.predict(x_val)

c = 0
for i in range(res.shape[0]):
    if(all(res[i]==y_val[i])):
        c+=1

c/res.shape[0]


#%%
np.array([1,2,3,4])==np.array([1,2,33,4])


#%%
np.round(0.7429,decimals=1)


#%%
y


#%%
a = np.random.rand(3,4)
a


#%%
a[0] = [1,2,3,4]


#%%
a


#%%
for i,sample in enumerate(X):
    print(i,sample)


#%%



#%%
import pandas as pd

train_dig = pd.read_csv('digit-recognizer/train.csv')


#%%
y = train_dig.pop('label')


#%%
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.array(y).reshape(-1,1))


#%%
train_dig = train_dig/255


#%%
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train_dig,y,test_size=0.2)


#%%
nn = littleNN(50)
nn.fit(x_train.values,y_train,epochs=100)


#%%
res = nn.predict(x_val.values)

c = 0
for i in range(res.shape[0]):
    if(all(res[i]==y_val[i])):
        c+=1

c/res.shape[0]


#%%
nn.w1


#%%



