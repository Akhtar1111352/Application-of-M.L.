#!/usr/bin/env python
# coding: utf-8

# CODE BY M M AKHTAR
#    50008492
# LOGISTIC REGRESSION

# In[71]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style = "white", color_codes = True)


# In[72]:


iris= pd.read_csv("IRIS.csv")
iris.head()


# In[73]:


iris["species"].value_counts()


# In[74]:


sns.FacetGrid(iris,hue = "species", height=6).map(plt.scatter, "petal_length", "sepal_width").add_legend()


# LOGISTIC REGRESSION

# In[75]:


flower_mapping = {'setosa' :0,'versicolor' : 1, 'virginica':2}


# In[76]:


iris.head()


# In[77]:


x= iris[['sepal_length','sepal_width','petal_length','petal_width']].values
y = iris[["species"]].values


# LOGISTIC REGRESSION

# In[78]:


from sklearn.linear_model import LogisticRegression


# In[79]:


model = LogisticRegression()


# In[80]:


model.fit(X,y)


# In[81]:


model.score(X,y)


# In[82]:


expected = y
predicted = model.predict(X)
predicted


# In[84]:


from sklearn import metrics
print(metrics.classification_report(expected, predicted))


# In[85]:


print(metrics.confusion_matrix(expected,predicted))


# In[93]:


model =   LogisticRegression(C=20,penalty = 'l2')


# In[94]:


model.fit(X,y)


# In[95]:


model.score(X,y)

