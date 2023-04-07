# CODE BY M M AKHTAR 
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


prices = pd.read_csv('prices.csv')
prices.head()

yahoo = prices[prices['symbol'] =='YHOO']

yahoo_prices = yahoo.close.values.astype("float32")

yahoo_prices = yahoo_prices.reshape(1762,1)

yahoo_prices.shape

plt.plot(yahoo_prices)

plt.show()

np.random.seed(7)

scaler = MinMaxScaler(feature_range = (0,1))

yahoo_prices = scaler.fit_transform(yahoo_prices)

train_size = int(len(yahoo_prices)*0.80)

train_size = len(yahoo_prices)- train_size

train ,test = yahoo_prices[0:train_size,:],yahoo_prices[train_size:len(yahoo_prices),:]

print(len(train),len(test))

def create_dataset(dataset,look_back = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back,0])
    return np.array(dataX),np.array(dataY)

look_back = 1
trainX,trainY = create_dataset(train,look_back)
testX,testY = create_dataset(test,look_back)
trainX=np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))
model = Sequential()
model.add(LSTM(4,input_shape=(1,look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer = 'adam')
model.fit(trainX,trainY,epochs = 100,batch_size = 1, verbose = 2)
trainPredict = model.predict(trainX)
testpredict = model.predict(testX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testpredict = scaler.inverse_transform(testpredict)
testY = scaler.inverse_transform([testY])
trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
testScore = math.sqrt(mean_squared_error(testY[0],testpredict[:,0]))
print('test Score : %.2f RMSE' % (testScore))
trainPredictPlot = np.empty_like(yahoo_prices)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back,:]=trainPredict
testpredictPlot = np.empty_like(yahoo_prices)
testpredictPlot[:,:]=np.nan
testpredictPlot[len(trainPredict)+(look_back*2)+1:len(yahoo_prices)-1,:]=testpredict
plt.plot(scaler.inverse_transform(yahoo_prices))
plt.plot(trainPredictPlot,label='True')
plt.plot(testpredictPlot,label='LSTM')
plt.show()

