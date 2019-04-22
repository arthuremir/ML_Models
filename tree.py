#%%
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

#%%
def check_purity(data):
    unique_values = np.unique(data, return_counts=True)[0]
    return len(unique_values) == 1

#%%
def find_splits(data):
    all_splits = []
    for j in range(data.shape[1]):
        splits = np.array([])
        unique_values = np.unique(data[:,j])
        for i in range(len(unique_values)-1):
            splits = np.append(splits, (unique_values[i] + unique_values[i+1]) / 2)
        all_splits.append(splits)
    return np.array(all_splits)
     
#%%
def cross_entropy(data):
    _, unique_count = np.unique(data, return_counts=True)
    probs = unique_count / len(data)
    return np.sum(probs* -np.log2(probs))

#%%
def overall_entropy(probs, cross_entropy):
    return np.sum(probs*cross_entropy)

#%%
def splits_entropy(X, y, potential_splits):
    all_entropy_list = []
    for j in range(X.shape[1]):
        entropy_list = np.array([])
        for i in potential_splits[j]:
            mask = X[:,j] <= i
            probs_masked = np.array([np.mean(mask), np.mean(~mask)])
            parts_cross_entropy = np.array([cross_entropy(y[mask]), cross_entropy(y[~mask])])
            entropy_list = np.append(entropy_list, overall_entropy(probs_masked, parts_cross_entropy))
        all_entropy_list.append(entropy_list)
    return np.array(all_entropy_list)

#%%
def determine_optimal_split(splits, splits_entropy):
    optimal_split = 999
    for i in range(len(splits)):
        min_entropy_for_col = np.min(splits_entropy[i])
        if min_entropy_for_col < optimal_split:
            optimal_split = min_entropy_for_col
            optimal_col = i
    return optimal_col, np.mean(splits[optimal_col][splits_entropy[optimal_col]==optimal_split])

#%%
""" plt.figure(figsize=(15,25))
for i in range(3):
    plt.subplot(6,1,i+1)
    plt.scatter(split_list[i], split_entropy_list[i])

for i in range(3):
    plt.subplot(6,1,i+4)
    plt.scatter(X[:,2], X[:,i], c=y)
    plt.vlines(split_value, np.min(X[:,2]), np.max(X[:,2]), colors='white') """

#%%
def classify(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    return unique_classes[np.argmax(class_counts)]


#%%
def decision_tree_classifier(X, y):
    print(X)
    print(y)

    #base case
    if check_purity(y):
        return classify(y)
    else:
        split_list = find_splits(X)
        split_entropy_list = splits_entropy(X, y, split_list)
        split_column, split_value = determine_optimal_split(split_list, split_entropy_list)

        criteria = '{0} <= {1}'.format(split_column, split_value)
        subtree = {criteria:[]}

        right_leaf = decision_tree_classifier(X[X[:,split_column]<=split_value], 
                                            y[X[:,split_column]<=split_value])
        left_leaf = decision_tree_classifier(X[X[:,split_column]>split_value], 
                                            y[X[:,split_column]>split_value])

        subtree[criteria].append(right_leaf)
        subtree[criteria].append(left_leaf)

        return subtree


#%%
decision_tree_classifier(X, y)

#%%
