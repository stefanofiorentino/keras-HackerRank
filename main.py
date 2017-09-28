

import pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from io import BytesIO
from zipfile import ZipFile
import sys
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
import datetime
#import pyflux as pf
import statsmodels.api as sm
from scipy import stats
from matplotlib import rcParams
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Activation
from pandas import concat
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.layers import Dropout
from sklearn import preprocessing
def __main__():
    # va messo t , t-1 , t-2  .. fare il test set su t ,   e il training su t-2 , y  = t-1
    df = pd.read_csv('/HospitalRevenue.csv')  
    
    encoder = preprocessing.LabelEncoder()
    df['Hospital_ID'] = encoder.fit_transform(df['Hospital_ID'])
    df['Region_ID'] = encoder.fit_transform(df['Region_ID'])
    df['District_ID'] = encoder.fit_transform(df['District_ID'])
    df['Instrument_ID'] = encoder.fit_transform(df['Instrument_ID'])
    
    df.drop( ['Year Total'], axis=1,inplace=True)  # remove of the year total from dataframe
    # transform as stack     
    df = df.set_index(['Hospital_ID','Region_ID','District_ID','Instrument_ID']) 
    stackedTrain = df.stack()  
    stackedTrain = stackedTrain.to_frame().reset_index() 
    stackedTrain.drop( ['level_4'], axis=1,inplace=True) 
    
    df_train=stackedTrain.rename(columns = {0:'value_t'}) 
    
    df_train['value_t_1'] = df_train.groupby(['Hospital_ID','Region_ID', 'District_ID'])['value_t'].transform(lambda x: x.shift())
    
    df_train['value_t_2'] = df_train.groupby(['Hospital_ID','Region_ID', 'District_ID'])['value_t'].transform(lambda x: x.shift(2))
    
    # definisci la variabile y
    y_train = df_train['value_t_1']
    y_test = df_train['value_t']
    
    x_test = df_train.copy()
    x_train = df_train.copy()
    #rimuovi dal test tengo t-1 e t = y
    x_test.drop(['value_t'], axis=1,inplace=True)
    x_test.drop(['value_t_2'], axis=1,inplace=True)
    x_test.dropna()  
    # rimuovi y dal training set x  tengo t-2 e t-1 = y
    x_train.drop(['value_t'], axis=1,inplace=True)
    x_train.drop(['value_t_1'], axis=1,inplace=True) 
    x_train.dropna()
    
    y_train.dropna()
    y_test.dropna()
    
    #### transform to numpy
    x_train = x_train.values
    x_test = x_test.values
    
    look_back = 1
    
    trainY = y_train.values
    testY = y_test.values
    
    #
    
    
    trainX = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    testX = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1,1, x_train.shape[1]),stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(1,1, x_train.shape[1]) , stateful=True))
    model.add(Dense(64, activation='relu', init='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax')) 
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    
__main__();
