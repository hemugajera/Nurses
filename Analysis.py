#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install ruptures


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os
from datetime import date, datetime, timedelta


from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import ruptures as rpt


# In[3]:


# the Pandas library and is used to read a CSV file into a DataFrame
df = pd.read_csv("./cleaned_dataset.csv")


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


df['Stress_Level'].dtype


# In[9]:


df['datetime'] = pd.to_datetime(df['datetime'])


# In[10]:


# This will identifying monthly trend  
df['datetime'].dt.to_period('M')


# In[11]:


df['id']  = df['id'].astype(str)


# In[12]:


from sklearn import preprocessing
  
# Create an instance of the LabelEncoder class
label_encoder = preprocessing.LabelEncoder()
  
# Use the LabelEncoder to encode the "id" column as numerical values
df['id']= label_encoder.fit_transform(df['id'])


# In[13]:


df.info()


# In[14]:


# Plot a histogram of the "Stress_Level" column of the DataFrame
plt.hist(df['Stress_Level'])
# Display the plot
plt.show()


# In[15]:


# Set the size of the figure
plt.figure(figsize=(20,9))

# Generate the heatmap plot of the correlation matrix
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
  
# Display the plot
plt.show()


# In[16]:


df.info()


# In[17]:


# Drop the "datetime" column from the DataFrame
df = df.drop(['datetime'], axis=1)


# In[18]:


from sklearn.model_selection import train_test_split   # spliting the dataset
# Define the predictors and target variables
predictors=df.drop("Stress_Level",axis=1)  
target=df["Stress_Level"]

# Split the data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[19]:


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)


# In[27]:


from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
lr = LogisticRegression()

# Fit the model to the training data
lr.fit(X_train,y_train)


# In[21]:


# Use the trained model to make predictions on the test data
y_pred_test = lr.predict(X_test)

# Print the predicted values
y_pred_test


# In[22]:


from sklearn.metrics import accuracy_score

# Calculate the accuracy score of the model on the test data
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# In[29]:


from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_val_predict
# Initialize the logistic regression model
log_reg = LogisticRegression()

# Use k-fold cross-validation to evaluate model performance
scores = cross_val_score(log_reg, predictors, target, cv=5)
y_pred = cross_val_predict(log_reg, predictors, target, cv=5)

# Calculate and print the accuracy score of the model
accuracy = accuracy_score(target, y_pred)
print(f"Accuracy: {accuracy:.3f}")


# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


df_lag = pd.read_csv('./cleaned_dataset.csv')
train_set = df_lag.iloc[:,0:48]
labels = df_lag.iloc[:,48:49]

#Create a random forest Classifier
clf = RandomForestClassifier(n_estimators=100,max_depth=15)

# Split our data
train, test, train_labels, test_labels = train_test_split(predictors, target, test_size=0.50, random_state=20)
# predictors=df_lag.drop("Stress_Level",axis=1)  
# target=df_lag["Stress_Level"]


# train,test,train_labels,test_labels = train_test_split(predictors,target,test_size=0.20,random_state=0)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(train, train_labels.values.ravel())



# In[35]:


y_pred = clf.predict(test)

f1score   = f1_score        (test_labels, y_pred, average = 'macro')
recall    = recall_score    (test_labels, y_pred, average = 'macro')
precision = precision_score (test_labels, y_pred, average = 'macro')
accuracy  = accuracy_score  (test_labels, y_pred)

print('acc =', accuracy)
print('pre =', precision)
print('recall =', recall) 
print('f1 =', f1score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




