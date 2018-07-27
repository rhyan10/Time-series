# load and plots
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.regularizers import L1L2
from math import sqrt
import matplotlib

# be able to save images on server
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy


# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)  # puts data into pandas data frame
    columns = [df.shift(i) for i in range(1, lag + 1)]  # does something regarding shifting data
    columns.append(df)  # puts the shifted data into the data frame
    df = concat(columns, axis=1)  # concatenates on an axis: 0 - index; 1 - columns
    return df  # returns the dataframe


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):  # for loop loops through from 1 to the length of the dataset
        value = dataset[i] - dataset[i - interval]  # the value saved is the current value takaway the distance from the current value and the interval
        diff.append(value)  # appends the values into the list, populating the list one by one as it goes
    return Series(diff)  # returns the list in pandas series form e.g: MSFT 56.50         ;dtype: float64


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]  # yhat plus the last item in the history array


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))  # squeezes all values range of -1 and 1, add robustness to small standard deviations
    scaler = scaler.fit(train)  # computes the min ma to be used later for scaling
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])  # reshapes the train array into the shape of 0 and 1
    train_scaled = scaler.transform(train)  # does the transformation???
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])  # does the same thing but to the test data
    test_scaled = scaler.transform(test)  # does the same thing but to the test data
    return scaler, train_scaled, test_scaled  # returns the sacaler, train and test variables


# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]  # creates new row with everything in X plus yhat
    array = numpy.array(new_row)  # creates a numpy array with the new row
    array = array.reshape(1, len(array))  # reshapes the array form 1 to the length of the array
    inverted = scaler.inverse_transform(array)  # inverts the array
    return inverted[0, -1]  # returns all values between 0 and -1


# fit an LSTM network to training data
def fit_lstm(train, n_batch, nb_epoch, n_neurons):
    X, y = train[:, 0:-1], train[:, -1]  # X is assigned by multi-dimensional indexing
    X = X.reshape(X.shape[0], 1, X.shape[1])  # reshapes X into its 3 dimensional form
    model = Sequential()  # calls sequential LSTM model
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True)) # Adds LSTM
    model.add(Dense(1))  # adds 1 dense layer
    model.compile(loss='mean_squared_error', optimizer='adam')  # compiles it
    for i in range(nb_epoch):  # repetitions
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)  # trains the model (fits)
        model.reset_states()  # used with stateful, make consecutive model calls independant
    return model


# run a repeated experiment
def experiment(series, n_lag, n_repeats, n_epochs, n_batch, n_neurons):
    # transform data to be stationary
    raw_values = series.values  # puts the series values into a values variable with just values without series
    diff_values = difference(raw_values, 1)  # find the difference between the raw value and 1
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, n_lag)  # converts time series to supervised
    supervised_values = supervised.values[n_lag:, :]  # Gets all the values from the lag?
    # split data into train and test-sets
    train, test = supervised_values[0:-12], supervised_values[-12:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(n_repeats):
        # fit the model
        train_trimmed = train_scaled[2:, :]
        lstm_model = fit_lstm(train_trimmed, n_batch, n_epochs, n_neurons)
        # forecast test dataset
        test_reshaped = test_scaled[:, 0:-1]
        test_reshaped = test_reshaped.reshape(len(test_reshaped), 1, 1)
        output = lstm_model.predict(test_reshaped, batch_size=n_batch)
        predictions = list()
        for i in range(len(output)):
            yhat = output[i, 0]
            X = test_scaled[i, 0:-1]
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
            # store forecast
            predictions.append(yhat)
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
        print('%d) Test RMSE: %.3f' % (r + 1, rmse))
        error_scores.append(rmse)
    return error_scores


# configure the experiment
def run():
    # load dataset
    series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    # configure the experiment
    n_lag = 1
    n_repeats = 30
    n_epochs = 1000
    n_batch = 4
    n_neurons = 3
    # run the experiment
    results = DataFrame()
    results['results'] = experiment(series, n_lag, n_repeats, n_epochs, n_batch, n_neurons)
    # summarize results
    print(results.describe())
    # save boxplot
    results.boxplot()
    pyplot.savefig('experiment_baseline.png')


# entry point
run()

