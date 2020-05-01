# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:26:35 2020
SVM Model

@author: Dhairya
"""

#Importing the Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load the Cancer data
cell_df = pd.read_csv("cell_samples.csv")
cell_df.head()
cell_df.info()

#Lets look at the distribution of the classes based on Clump thickness and Uniformity of cell size
ax = cell_df[cell_df['Class'] == 4].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

#It looks like the BareNuc column includes some values that are not numerical. We can drop those rows:
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes
cell_df.info()

#setting the independent and dependent variable
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y [0:5]

#Splitting data into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

#Training the SVM model 
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

#Predicting the output
yhat = clf.predict(X_test)
yhat [0:5]

#Checking the accuracy
from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted')) 

from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, yhat))