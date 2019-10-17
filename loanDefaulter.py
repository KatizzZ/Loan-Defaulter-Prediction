#!/usr/bin/env python
# coding: utf-8

# Importing Pandas for data frame and Scikit-learn for machine learning library

# In[1]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Reading data into frames and showing

# In[2]:


data = pd.read_excel('case_study_data.xlsx')
data.head()


# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.isnull().values.any()


# In[6]:


data[data.isnull().any(axis=1)] 


# Add a column to insert row numbers for entire dataframe for easy sampling of observations

# In[7]:


data ['a'] = pd.DataFrame({'a':range(1001)})
data.columns


# Selecting every 10th row of the data to train the model

# In[8]:


sampled_df = data[(data['a'] % 10) == 0]
sampled_df.shape


# Remaing data for testing to check the results

# In[9]:


sampled_df_remaining = data[(data['a'] % 10) != 0]
sampled_df_remaining.shape


# In[10]:


y = sampled_df['status'].copy()


# In[11]:


loan_features = ['duration','amount','inst_rate', 'age', 'num_credits','dependents']


# In[12]:


x = sampled_df[loan_features].copy()


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)


# In[14]:


loan_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
loan_classifier.fit(X_train, y_train)


# In[15]:


predictions = loan_classifier.predict(X_test)
predictions[:20]


# In[16]:


accuracy_score(y_true = y_test, y_pred = predictions)


# In[17]:


X1 = sampled_df_remaining[loan_features].copy()


# In[18]:


y1 = sampled_df_remaining['status'].copy()


# In[19]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=324)


# In[20]:


loan_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
loan_classifier.fit(X1_train, y1_train)


# In[21]:


predictions1 = loan_classifier.predict(X1_test)


# In[22]:


predictions1[:20]


# In[23]:


accuracy_score(y_true = y1_test, y_pred = predictions1)
