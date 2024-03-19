#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[4]:


# df=pd.read_excel("/content/drive/My Drive/flight fare pediction/Data_Train.xlsx")
df = pd.read_excel("Data_train.xlsx")
df.head(10)


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.dropna(inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


#unique values and eccoding 
df.Total_Stops.unique()


# In[11]:


df.Total_Stops.value_counts()


# In[12]:


df.Source.unique()


# In[13]:


df.Destination.unique()


# In[14]:


df.Airline.unique()


# In[15]:


df.Additional_Info.unique()


# In[16]:


df['Additional_Info'].value_counts()


# In[17]:


df.drop('Additional_Info',axis=1,inplace=True)


# In[18]:


df.head()


# In[19]:


#label encoding
from sklearn.preprocessing import LabelEncoder


# In[20]:


encoder = LabelEncoder()


# In[21]:


encoder.fit(df.Airline)


# In[22]:


df['Airline'] = encoder.transform(df.Airline)


# In[23]:


encoder.classes_


# In[24]:


df['Airline'].unique()


# In[26]:


df['Source'] = encoder.fit_transform(df['Source'])


# In[27]:


encoder.classes_


# In[28]:


df['Destination'] = encoder.fit_transform(df['Destination'])


# In[29]:


encoder.classes_


# In[30]:


df.columns


# In[31]:


df.Total_Stops.unique()


# In[32]:


df['Total_Stops'] = df.Total_Stops.apply(lambda x:'0 stop' if x=='non-stop' else x)


# In[33]:


df['Total_Stops'] = df.Total_Stops.apply(lambda x:int(x.split()[0]))


# In[34]:


df.head()


# In[35]:


#working with time
df['Date_of_Journey'] =pd.to_datetime(df['Date_of_Journey'])


# In[36]:


df.info()


# In[48]:


df['Dep_Time'] = df.Dep_Time.apply(lambda x:int(x.split(':')[0]))


# In[49]:


df['Arrival_Time'] = df.Arrival_Time.apply(lambda x:int(x.split(':')[0]))


# In[50]:


df['Arrival_Time'] = df[['Dep_Time','Arrival_Time']].apply(lambda x:x['Arrival_Time']+24 if x['Dep_Time']>x['Arrival_Time']


# In[51]:


df.head()


# In[52]:


df['Month_of_Journey'] =  df['Date_of_Journey'].map(lambda x:x.month)


# In[53]:


def duration_time(x):
    x = x.split()
    x = list(map(lambda t:int(t[:-1]),x))
    if len(x) == 1:
        return x[0]*60
    else:
        return x[0]*60 + x[1]


# In[54]:


df['Duration'] =  df['Duration'].apply(duration_time)


# In[55]:


df.head()


# In[56]:


#target visualization
sns.distplot(df['Price'],kde=False)


# In[57]:


df[df['Price']>40000]


# In[58]:


df.drop(df[df['Price']>40000].index,inplace=True)


# In[59]:


sns.distplot(df['Price'],kde=False)


# In[60]:


#input for model
df._get_numeric_data().head()


# In[61]:


X = df._get_numeric_data().drop('Price',axis=1)


# In[62]:


y = df['Price']


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[65]:


#model building
from sklearn.linear_model import LinearRegression


# In[66]:


model  = LinearRegression()


# In[67]:


model.fit(X_train,y_train)


# In[68]:


predictions = model.predict(X_test)


# In[71]:


#prediction visualization
sns.scatterplot(y_test,predictions)


# In[72]:


sns.distplot((y_test-predictions),bins=50,kde=False)


# In[73]:


#error calculation
from sklearn import metrics


# In[74]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


#output
#output contain price above the 35000 so its not bad prediction
#This model is predicting values of the fare with an error of 2400 rs.

