#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
x_axis = np.linspace(0,1,1000)+5*np.random.randn(1000)
y_axis = 5*x_axis + 4 + 10*np.random.randn(1000)

#%%
data = np.c_[x_axis, y_axis]
plt.scatter(data[:,0], data[:,1], s=10)

#%%
class LinearRegression():

    def __init__(self):
        pass

    def fit(self, X, y):
        X = np.c_[X, [1]*X.shape[0]]
        n_samples, n_features = X.shape
        
        self.w = np.random.randn(n_features)
        self.learning_curve = []
        self.w_list = []

        for sample, y_true in zip(X, y):
            y_predicted = sample * self.w
            error = y_predicted - y_true
            J = np.sum(error**2) / (2 * n_samples)
            self.learning_curve.append(J)
            self.w_list.append(np.copy(self.w))
            self.w -= np.sum(error * sample) / n_samples

    def predict(self, X):
        X = np.c_[X, [1]*X.shape[0]]
        return np.sum(X*self.w, axis=1)

    def plot_learning_curve(self):
        plt.plot(self.learning_curve)
        

#%%
lr = LinearRegression()
lr.fit(data[:,0].reshape(-1,1), data[:,1])

#%%
plt.figure(figsize=(20,10))
plt.scatter(data[:,0], data[:,1], s=10)
ax = plt.axes()
for w in lr.w_list:
    ax.plot(x_axis, x_axis*w[0]+w[1], c='white')

#%%
lr.plot_learning_curve()

#%%
lr.w_list

#%%
