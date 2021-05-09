#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor


# In[13]:


data = pd.read_csv('house_rental_data.csv.txt')
data.head()


# In[14]:


data.isnull().sum()


# In[15]:


data.info()


# In[16]:


sns.pairplot(data)    # we can observe here relation between price and all other attributes


# In[26]:


from sklearn.model_selection import train_test_split


# In[18]:


X = data.iloc[:,1:-1].values
Y = data.iloc[:,-1].values
print(X.shape)
Y.shape


# In[19]:


model = KNeighborsRegressor(n_neighbors=5)


# In[27]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state = 1)
model.fit(x_train,y_train)


# In[45]:


for i in range(1,25):
    model = KNeighborsRegressor(n_neighbors=i)
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state = 1)
    model.fit(x_train,y_train)
    print(f' for k = {i}  score =  {model.score(x_test,y_test)}')

    


# In[ ]:


# for k = 3 score is good or for k = 23 score is better


# In[ ]:





# In[ ]:





# In[ ]:




