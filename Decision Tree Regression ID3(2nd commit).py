#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv("CarPrice.csv")


# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.isnull().sum().sort_values(ascending=False)


# # Data Cleaning

# In[8]:


data.info()


# In[9]:


X = data.drop('price', axis=1)
y = data['price']


# In[10]:


# Split the data into train and test sets
train_size = int(0.8 * len(X))
train_X, test_X = np.split(X, [train_size])
train_y, test_y = np.split(y, [train_size])


# # Node class

# In[11]:


class Node:
   
    def __init__(self, feature=None, threshold=None, value=None, var_red=None, left=None, right=None):
        
         # for decision node
        self.feature = feature      # feature to split on
        self.threshold = threshold  # threshold value to split on
        self.left = left            # left subtree
        self.right = right          # right subtree
        self.var_red = var_red      # variance reduction
        # for leaf node
        self.value = value          # predicted value 


# # Tree Class 

# In[12]:


class DecisionTreeRegressor:
    
    def __init__(self, min_samples_split=2, max_depth=3):
        
     # initialize the root of the tree 
        self.root = None
        
     # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    # recursive function to build the tree
    def build_tree(self, X,y , curr_depth=0):       
        num_samples, num_features = X.shape
        num_classes=len(np.unique(y))
        
     # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            
     # find the best split
            best_split = self.find_best_split(dataset, num_samples, num_features)
            
            # check if SDR is positive or not
            if best_split["var_red"]>0:
                
            # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                
            # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                
            # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        
        # return leaf node
        return Node(value=leaf_value)
    
    


# In[13]:


# To find the best split
def find_best_split(self, dataset, num_samples, num_features):
        
        # To store the best split
        best_split = {}
        max_var_red = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
        # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    
                # compute SDR
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    
                    # update the best split if needed
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split
    
    # splitting the data 
def split(self, dataset, feature_index, threshold):
    
    dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
    dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
    return dataset_left, dataset_right
    
    


# In[14]:


# computing the variance reduction
def variance_reduction(self, parent, l_child, r_child):
    weight_l = len(l_child) / len(parent)
    weight_r = len(r_child) / len(parent)
    reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
    return reduction
    
def calculate_leaf_value(self, Y):
    val = np.mean(Y)
    return val
                
def print_tree(self, tree=None, indent=" "):
    if not tree:
        tree = self.root

    if node.value is not None:
        print(tree.value)

    else:
        print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
        print("%sleft:" % (indent), end="")
        self.print_tree(tree.left, indent + indent)
        print("%sright:" % (indent), end="")
        self.print_tree(tree.right, indent + indent)
    


# In[15]:


# function to train the tree
def fit(self, X, Y):
    self.tree = self.build_tree(X,y)
     
# function to predict new dataset
def make_prediction(self, x, tree):
    if tree.value!=None: return tree.value
    feature_val = x[tree.feature_index]
    if feature_val<=tree.threshold:
        return self.make_prediction(x, tree.left)
    else:
        return self.make_prediction(x, tree.right)
  
 # function to predict a single data point
def predict(self, X):
    predictions = [self.make_prediction(x, self.root) for x in X]
    return predictions


# In[16]:


dt_pred = Regressor.predict(test_X)


# In[ ]:





# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


num_attr=data.select_dtypes(['int64']).columns  
num_attr


# In[19]:


cat_attr = data.select_dtypes('object').columns
cat_attr


# In[26]:


plt.bar(data['enginetype'], data['car_ID'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




