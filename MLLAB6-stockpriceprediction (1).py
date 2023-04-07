#!/usr/bin/env python
# coding: utf-8

# # CODE BY M M AKHTAR
# 

# In[1]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.preprocessing import MinMaxScaler


# In[2]:


from sklearn.metrics import mean_squared_error,r2_score


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


prices = pd.read_csv('prices.csv')


# In[6]:


prices.head()


# In[7]:


yahoo = prices[prices['symbol'] =='YHOO']


# In[8]:


yahoo_prices = yahoo.close.values.astype("float32")


# In[9]:


yahoo_prices = yahoo_prices.reshape(1762,1)


# In[10]:


yahoo_prices.shape


# In[11]:


plt.plot(yahoo_prices)


# In[12]:


plt.show()


# In[13]:


np.random.seed(7)


# In[14]:


scaler = MinMaxScaler(feature_range = (0,1))


# In[15]:


yahoo_prices = scaler.fit_transform(yahoo_prices)


# In[16]:


train_size = int(len(yahoo_prices)*0.80)


# In[17]:


train_size = len(yahoo_prices)- train_size


# In[18]:


train ,test = yahoo_prices[0:train_size,:],yahoo_prices[train_size:len(yahoo_prices),:]


# In[19]:


print(len(train),len(test))


# In[20]:


def create_dataset(dataset,look_back = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back,0])
    return np.array(dataX),np.array(dataY)


# In[21]:


look_back = 1
trainX,trainY = create_dataset(train,look_back)
testX,testY = create_dataset(test,look_back)


# In[22]:


trainX=np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))


# In[23]:


testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))


# In[24]:


model = Sequential()


# In[25]:


model.add(LSTM(4,input_shape=(1,look_back)))


# In[26]:


model.add(Dense(1))


# In[27]:


model.compile(loss='mean_squared_error',optimizer = 'adam')


# In[28]:


model.fit(trainX,trainY,epochs = 100,batch_size = 1, verbose = 2)


# In[29]:


trainPredict = model.predict(trainX)


# In[30]:


testpredict = model.predict(testX)


# In[31]:


trainPredict = scaler.inverse_transform(trainPredict)


# In[32]:


trainY = scaler.inverse_transform([trainY])


# In[33]:


testpredict = scaler.inverse_transform(testpredict)


# In[34]:


testY = scaler.inverse_transform([testY])


# In[35]:


trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))


# In[36]:


testScore = math.sqrt(mean_squared_error(testY[0],testpredict[:,0]))


# In[37]:


print('test Score : %.2f RMSE' % (testScore))


# In[38]:


trainPredictPlot = np.empty_like(yahoo_prices)


# In[39]:


trainPredictPlot[:,:]=np.nan


# In[40]:


trainPredictPlot[look_back:len(trainPredict)+look_back,:]=trainPredict


# In[41]:


testpredictPlot = np.empty_like(yahoo_prices)


# In[42]:


testpredictPlot[:,:]=np.nan


# In[43]:


testpredictPlot[len(trainPredict)+(look_back*2)+1:len(yahoo_prices)-1,:]=testpredict


# In[44]:


plt.plot(scaler.inverse_transform(yahoo_prices))
plt.plot(trainPredictPlot,label='True')
plt.plot(testpredictPlot,label='LSTM')
plt.show()

