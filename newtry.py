from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(filename, seq_len):

    df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv', dtype=float, header=None).values
    df[df == 0] = 1
    data = df.tolist()
    
    result = []
    for index in range(len(data) - seq_len):
        result.append(data[index: index + seq_len])
    
    #normalize data
    d0 = np.array(result)
    dr = np.zeros_like(d0)
    dr[:,1:,:] = d0[:,1:,:] / d0[:,0:1,:] - 1
    result = dr[:, :, :6] #only want six columns since our dataset has extraneous information
    
    #to unnormalize it later
    start = int(dr.shape[0]*0.9)
    end = int(dr.shape[0]+1)
    unnormalize_bases = d0[start:end,0:1,0]

    #split into train and testing data
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    y_train = y_train[:,3]
   
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, 49, :]
    y_test = y_test[:,3]
    
    #the day before y_test aka today's price
    y_dayb4 = result[int(row):, 48, :]
    y_dayb4 = y_dayb4[:,3]
    
    return [x_train, y_train, x_test, y_test, unnormalize_bases, y_dayb4]

sequence_length = 50
lookback = sequence_length - 1
X_train, y_train, X_test, y_test, unnormalize_bases, y_dayb4 = load_data('bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv', sequence_length)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

model = Sequential()

model.add(Bidirectional(LSTM(lookback,return_sequences=True), input_shape=(lookback,X_train.shape[-1]),))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM((lookback*2), return_sequences=True)))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(lookback, return_sequences=False)))

model.add(Dense(output_dim=1))
model.add(Activation('tanh'))

model.compile(loss='rmse', optimizer='adam')
print(model.summary())

start = time.time()
model.fit(X_train, y_train, batch_size=1024, nb_epoch=10, validation_split=0.05)
print('training time : ', time.time() - start)

y_predict = model.predict(X_test)
real_y_test = np.zeros_like(y_test)
real_y_predict = np.zeros_like(y_predict)

for i in range(y_test.shape[0]):
    y = y_test[i]
    predict = y_predict[i]
    real_y_test[i] = (y+1)*unnormalize_bases[i]
    real_y_predict[i] = (predict+1)*unnormalize_bases[i]
    
plt.plot(real_y_predict, color='green')
plt.plot(real_y_test, color='blue')
plt.title("Bitcoin Price in 95-Day Period, 07/09/2017-10/12/2017")
plt.ylabel("Price (USD)")
plt.xlabel("Time (Days)")
plt.savefig('price.png', dpi = 600)
plt.show()

print("y_predict:", y_predict.shape)
y_dayb4 = np.reshape(y_dayb4, (-1, 1))
print("y_dayb4:", y_dayb4.shape)
y_test = np.reshape(y_test, (-1, 1))
print("y_test:", y_test.shape)
delta_predict = y_predict - y_dayb4
delta_real = y_test - y_dayb4
plt.plot(delta_predict, color='green')
plt.plot(delta_real, color='blue')
plt.title("Bitcoin Price % Change in 95-Day Period, 07/09/2017-10/12/2017")
plt.ylabel("Percent Increase")
plt.xlabel("Time (Days)")
plt.savefig('best_fluctuation.png', dpi = 600)
plt.show()

#one means predict stock increases in price, zero means decrease
delta_predict_1_0 = np.empty(delta_predict.shape)
delta_real_1_0 = np.empty(delta_real.shape)

for i in range(delta_predict.shape[0]):
    if delta_predict[i][0] > 0:
        delta_predict_1_0[i][0] = 1
    else:
        delta_predict_1_0[i][0] = 0

for i in range(delta_real.shape[0]):
    if delta_real[i][0] > 0:
        delta_real_1_0[i][0] = 1
    else:
        delta_real_1_0[i][0] = 0    

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for i in range(delta_real_1_0.shape[0]):
    real = delta_real_1_0[i][0]
    predicted = delta_predict_1_0[i][0]
    if real == 1:
        if predicted == 1:
            true_pos += 1
        else:
            false_pos += 1
    elif real == 0:
        if predicted == 0:
            true_neg += 1
        else:
            false_neg += 1

print("true_pos:", true_pos)
print("true_neg:", true_neg)
print("false_pos:", false_pos)
print("false_neg:", false_neg)

precision = float(true_pos) / (true_pos + false_pos)
recall = float(true_pos) / (true_pos + false_neg)
F1 = 2.0 / (1/precision + 1/recall)

print()
print("precision:", precision)
print("recall:", recall)
print("F1:", F1)

#MSE
from sklearn.metrics import mean_squared_error
print()
print("Testing MSE:", np.sqrt(mean_squared_error(y_predict.flatten(), y_test.flatten())))



# add one additional data point to align shapes of the predictions and true labels
X_test = np.append(X_test, scaler.transform(working_data.iloc[-1][0]))
X_test = np.reshape(X_test, (len(X_test), 1, -1))
#X_test = scale(X_test)

# get predictions and then make some transformations to be able to calculate RMSE properly in USD
prediction = model.predict(X_test)
prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
prediction2_inverse = np.array(prediction_inverse[:,0][1:])
Y_test2_inverse = np.array(Y_test_inverse[:,0])

trace1 = go.Scatter(
    x = np.arange(0, len(prediction2_inverse), 1),
    y = prediction2_inverse,
    mode = 'lines',
    name = 'Predicted labels',
    line = dict(color=('rgb(244, 146, 65)'), width=2)
)
trace2 = go.Scatter(
    x = np.arange(0, len(Y_test2_inverse), 1),
    y = Y_test2_inverse,
    mode = 'lines',
    name = 'True labels',
    line = dict(color=('rgb(66, 244, 155)'), width=2)
)

data = [trace1, trace2]
layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted',
             xaxis = dict(title = 'Day number'), yaxis = dict(title = 'Price, USD'))
fig = dict(data=data, layout=layout)
py.plot(fig, filename='results_demonstrating0')

