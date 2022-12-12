# -*- coding: utf-8 -*-
"""
Yanni Tsetsekos
MEM T680 HW 3
"""
## Here are some packages and modules that you will use. Make sure they are installed.

# for basic operations
import numpy as np 
import pandas as pd 

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# for modeling 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

#from google.colab import drive
#drive.mount('/content/drive')
#import sys
#sys.path.insert(0,'/content/drive/My Drive/ColabNotebooks')
import plotly.express as px

from imblearn.over_sampling import SMOTE

# to avoid warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Loading the data
dataset = pd.read_csv('uci-secom.csv') 
print("Shape of data:", dataset.shape) # Starting shape of dataset
print(dataset.head())

# Filtering Data

# to visualize where data is missing and shows up as NaN/null:
missing_values = dataset.isnull()
sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis') # plot missing values in lighter color
# dataset.dropna(axis=1, inplace=True) # Removes cols with null values

nan_per_feature = dataset.isna().sum() # null entries per feature
nan_per_feature_hist = [] # histogram list
for ind, i in enumerate(dataset.columns): 
    for j in range(nan_per_feature[ind]): 
        nan_per_feature_hist.append(i) # add to histogram list

px.histogram(data_frame=nan_per_feature_hist, title="Null values per feature") # histogram of the null values per feature

# Removing Sparse Features
preDropShape = dataset.shape
count_dropped = 0 # counter for dropped features
for row, col in enumerate(dataset.columns):
    num_nan = dataset[col].isna().sum() # adds up number of elements that are null
    if num_nan > 100: # if missing more than 100 entries
        dataset.drop(col, inplace=True, axis=1) # disregards data that is considered sparse
        drop_count += 1 # Increment drop feature counter

dataset.drop("Time", inplace=True, axis=1) # drops the Time feature
drop_count += 1
print("# Features Dropped:", drop_count) 
filtered_shape = dataset.shape 

# compare shapes
print("Pre-feature drop dataframe shape:", preDropShape)
print("Post-feature drop dataframe shape:", dataset.shape)

num_row_missing = 0  # num rows with missing values
for i, row in enumerate(dataset.index):
    if dataset.loc[i,:].isna().sum().sum() > 0:
      
        num_row_missing += 1
        dataset.drop(labels=row, inplace=True, axis=0) # drops the rows that contain any amount of missing data

print("Number of rows in data with missing data values:", num_row_missing)
print("Final shape of data:", dataset.shape)