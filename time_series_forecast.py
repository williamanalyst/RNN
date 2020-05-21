""" Stock Price forecasting with RNN - LSTM:
    # 1. f(n) = g(n-1, n-2, n-3, ... , n-60), therefore forecast values actually used both train and test data.
    # 3. test dataset is actually a combination of original train + test dataset, which will be concated and then inserted into model.
"""
#
# In[]: 
import pandas as pd
import os
os.chdir('C:\Python_Project\Deep_Learning\Data')
import numpy as np # only arrays are allowed in the rnn model
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc= {'figure.figsize':(12, 6)}, style = 'whitegrid', palette = 'Set1')
#
dataset_train = pd.read_excel('historical_wine_pricing.xlsx', sheet_name = 'price_data').iloc[:-30, :]
training_set = dataset_train.iloc[:, 18:19].values # use x:y to create a numpy array rather than single vector
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler # 
sc = MinMaxScaler(feature_range = (0, 1)) # default == (0, 1)
training_set_scaled = sc.fit_transform(training_set) # 
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
# np.shape(dataset_train)[0]
for i in range(60, np.shape(dataset_train)[0]): # 
    X_train.append(training_set_scaled[i-60: i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train) #
#
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # Reshaping to RNN accrpted array format
# Building the RNN
from keras.models import Sequential # Importing the Keras libraries and packages
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
# 
regressor = Sequential() # Initialising the RNN
#
regressor.add(LSTM(units = 50, 
                   return_sequences = True, 
                   input_shape = (X_train.shape[1], 1))) # Adding the first LSTM layer and some Dropout regularisation
regressor.add(Dropout(0.2))
# 
regressor.add(LSTM(units = 50, return_sequences = True)) # Adding a second LSTM layer and some Dropout regularisation
regressor.add(Dropout(0.2))
# 
regressor.add(LSTM(units = 50, return_sequences = True)) # Adding a third LSTM layer and some Dropout regularisation
regressor.add(Dropout(0.2))
# 
regressor.add(LSTM(units = 50)) # Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(Dropout(0.2)) # 
# 
regressor.add(Dense(units = 1)) # Adding the output layer
# 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # Compiling the RNN
# 
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) # Fitting the RNN to the Training set
#
regressor.save('test_rnn.h5') #
#
from keras.models import load_model #
model2 = load_model('test_rnn.h5')
model2.summary()
##
# In[]:
#
# Making the predictions and visualising the results
dataset_test = pd.read_excel('historical_wine_pricing.xlsx', 
        sheet_name = 'price_data').iloc[-30:, :].iloc[:, 18:19].values # use 1:2 to create a numpy array rather than single vector
#
dataset_total = pd.concat((dataset_train.iloc[:, 18:19], 
                           pd.read_excel('historical_wine_pricing.xlsx', sheet_name = 'price_data').iloc[-30:, :].iloc[:, 18:19]),
                            axis = 0) # combining both train dataset and test dataset for predictions
relevant_periods = 60 # estimation based on 2 previous months (similar to ARIMA(60))
inputs = dataset_total[len(dataset_total) - len(dataset_test) - relevant_periods:].values # 
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # transform both test and train dataset as new inputs for testing
#
X_test = [] # input of the test-set
for i in range(relevant_periods, (relevant_periods + len(dataset_test))): # input for the test-set: (60 + number of outputs)
    X_test.append(inputs[i-relevant_periods: i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # 3-layer input (array) for the model
predicted_index = regressor.predict(X_test)
predicted_index = sc.inverse_transform(predicted_index) # 
#
#
real_index = dataset_test #
# Visualising the results
plt.plot(real_index, color = 'red', label = 'Real Index') #
plt.plot(predicted_index, color = 'blue', label = 'Predicted Index') # 
plt.title('ASX 200 Share Index')
plt.xlabel('Time')
plt.ylabel('Index Value')
plt.legend()
plt.show()
#
plt.plot(training_set, color = 'r', label = 'actual price')
plt.show()
