#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Importing the dataset
#Extracting the independent and dependent variables
Development = pd.read_csv("human_development.csv")

#Independent Variables
X = Development.iloc[:,[0,1,2,3,4,5,7]]

#Dependant Variable
y = Development.iloc[:,2].values

Development.head()


# In[3]:


#Fixing the error
X.columns=X.columns.str.strip()


# In[4]:


#Data Exploration
#Development Score

g = sns.stripplot(x="Country", y="Human Development Index (HDI)", data=Development, jitter=True)
plt.xticks(rotation=90)


# In[5]:


#Data Exploration
#Life Expectancy at Birth

g = sns.stripplot(x="Country", y="Life Expectancy at Birth", data=Development, jitter=True)
plt.xticks(rotation=90)


# In[6]:


#Data Exploration
#Expected Years of Education

g = sns.stripplot(x="Country", y="Expected Years of Education", data=Development, jitter=True)
plt.xticks(rotation=90)


# In[7]:


#Data Exploration
#Mean Years of Education

g = sns.stripplot(x="Country", y="Mean Years of Education", data=Development, jitter=True)
plt.xticks(rotation=90)


# In[8]:


#Data Exploration
#GNI per Capita Rank Minus HDI Rank

g = sns.stripplot(x="Country", y="GNI per Capita Rank Minus HDI Rank", data=Development, jitter=True)
plt.xticks(rotation=90)


# In[9]:


#Data visualization

sns.distplot(Development['Human Development Index (HDI)'])


# In[10]:


#Building the correalation matrix

heat = Development.iloc[:,[0,1,2,3,4,5,7]]
sns.heatmap(heat.corr())


# In[11]:


#Dropping nans (missing values)
X=X.dropna()


# In[12]:


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
x=pd.get_dummies(X,columns=['Country'])
print(x.values)


# In[13]:


#data and label
c=list(x.columns)
c.remove('HDI Rank')
y=x['HDI Rank']
x=x[c]


# In[14]:


#data and label
x.shape,y.shape


# In[15]:


#Train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[16]:


#Linear Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train, y_train)


# In[17]:


#Train and test score
print("Train: ",reg.score(x_train,y_train))
print("Test: ",reg.score(x_test,y_test))


# In[18]:


y_pred=reg.predict(x_test)
print(y_pred)


# In[19]:


#calculating the coefficients 
print(reg.coef_)


# In[20]:


#Calculating the intercept 
print(reg.intercept_) 


# In[21]:


#calculating the R squared value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[22]:


#y_test Values
y_test


# In[23]:


#y_pred values
y_pred


# In[24]:


#Predicited Y Versus Testing Y
plt.scatter(y_test,y_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

