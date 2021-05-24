#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv('HR_comma_sep.csv.txt')
dataset.head()


# In[22]:


#dataset.iloc[:,1].notnull()   #records batata h


# In[23]:


def func(x):
    if x.number_project > 5:
        return x.left *2
    else:
        return x.left


# In[28]:


dataset.apply(func,axis=1)


# In[ ]:





# In[30]:


dataset.groupby(['left']).sum()


# In[31]:


dataset.groupby(['left']).promotion_last_5years.sum()


# In[ ]:





# In[ ]:





# In[3]:


dataset['left'].value_counts()


# In[4]:


dataset['promotion_last_5years'].value_counts()


# In[5]:


dataset['sales'].value_counts()


# In[6]:


dataset = pd.get_dummies(dataset,['promotion_last_5years','salary'],drop_first=True)


# In[7]:


dataset.head()


# In[8]:


y = dataset['left'].values
X = dataset.drop('left',axis = 1).values
y


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


sc = StandardScaler()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)


# In[13]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[14]:


x_test


# In[15]:


from sklearn.linear_model import LogisticRegression 


# In[23]:


model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
model.score(x_test,y_test)


# In[17]:


from sklearn.metrics import r2_score 


# In[22]:


r2_score(y_test,y_pred)              # why i got different score ??????


# In[ ]:





# In[ ]:





# In[1]:





# In[ ]:





# In[ ]:




