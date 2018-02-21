# --------------------------------------------------- #
# IMPORT STATEMENTS
#
# ORIGINALLY SOURCED FROM 
# https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-5-rnn_stock_prediction.py
#
#
# --------------------------------------------------- #
import tensorflow as tf
import numpy as np
import os
import matlab.engine


# disable that logging nonsense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# --------------------------------------------------- #
# SCALE DATA
# --------------------------------------------------- #

def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# --------------------------------------------------- #
# DEINE MODEL
# --------------------------------------------------- #
def model(X, Y):
    # build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    logits = tf.contrib.layers.fully_connected(
        outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output
    return logits 


# --------------------------------------------------- #
# TENSORFLOWWING!!!
# --------------------------------------------------- #

# connect to matlab
eng = matlab.engine.connect_matlab('MATLAB_1156')


# train Parameters
seq_length = 3 # 7
data_dim = 5
hidden_dim = 20 #10
output_dim = 1
learning_rate = 0.01
iterations = 7500

# Open, High, Low, Volume, Close
xy = np.loadtxt('web-stock.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # Close as label

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    # print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# print(dataY)
# quit()

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

print(" x shape : {}".format(X.get_shape().as_list()))
print(" y shape : {}".format(Y.get_shape().as_list()))

logits = model(X, Y)


# cost/loss
loss = tf.reduce_sum(tf.square(logits - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        if ((i+1)%250 == 0):
            print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(logits, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))


    eng.workspace['testY'] = matlab.double( testY.tolist() ) 
    eng.workspace['test_predict'] = matlab.double( test_predict.tolist() ) 

    # # Plot predictions
    # plt.plot(testY)
    # plt.plot(test_predict)
    # plt.xlabel("Time Period")
    # plt.ylabel("Stock Price")
    # plt.show()