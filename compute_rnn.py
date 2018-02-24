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

saver_dir  = os.getcwd()
ckpt_path  = os.path.join(saver_dir, 'model.ckpt')
saver_path = os.path.join(ckpt_path, 'saver')

# matlab session
SESSION_NAME = 'MATLAB_16344'

# --------------------------------------------------- #
# DEINE MODEL
# --------------------------------------------------- #
def model(X, Y, variables):
    # build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    print( outputs.get_shape().as_list() )
    # logits = tf.contrib.layers.fully_connected(
    #     outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

    logits  = tf.nn.xw_plus_b( outputs[:,-1], variables['w_1'], variables['b_1'])

    return logits 

# --------------------------------------------------- #
# DEINE VARIABLES
# --------------------------------------------------- #
def variable_data():
  out = {
    'gs'  : tf.Variable(0, name='global_step', trainable=False),
    'b_1' : tf.Variable(tf.random_normal(shape=[output_dim]), dtype=tf.float32),
    'w_1' : tf.Variable(tf.random_normal(shape=[hidden_dim, output_dim]), dtype=tf.float32), 
  }
  return out


# --------------------------------------------------- #
# TENSORFLOWWING!!!
# --------------------------------------------------- #

# connect to matlab
eng = matlab.engine.connect_matlab(SESSION_NAME)

# number of 'previous' input vectors packed into a single time step
seq_length = 15 # 7

# size of the INPUT vector associated with a single time step 
data_dim = 5

# size of the OUTPUT vector associated with a single time step 
output_dim = 1

# number of hidden states to keep track of
hidden_dim = 20 #10
learning_rate = 0.01
iterations = 2500

# dataX = np.asarray( dataX )
# dataY = np.asarray( dataY )
# print(dataX.shape)
# print(dataY.shape)
# (716, 15, 5)    => each of the 716 :) inputs has 15 time steps with 5 points each 
# (716, 1)        => each of the 716 :) outputs has 1 point
# quit()

# grab sequence from MATLAB
trainX = np.asarray( eng.workspace['trainX'] )
trainY = np.asarray( eng.workspace['trainY'] )
testX  = np.asarray( eng.workspace['testX'] )
testY  = np.asarray( eng.workspace['testY'] )


# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

variables = variable_data()
print(" x shape : {}".format(X.get_shape().as_list()))
print(" y shape : {}".format(Y.get_shape().as_list()))

logits = model(X, Y, variables)

# cost/loss
loss = tf.reduce_sum(tf.square(logits - Y))  # sum of the squares
## this appears to do the same thing, but is not the same thing
## loss = tf.losses.mean_squared_error(labels=Y, predictions=logits)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss, global_step=variables['gs'])

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

#setup saver
saver = tf.train.Saver(max_to_keep=3)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(saver_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('saver restored')

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

    # save the saver
    saver.save(sess, ckpt_path, global_step=variables['gs'])

    # this will set the MATLAB variables
    eng.workspace['testY'] = matlab.double( testY.tolist() ) 
    eng.workspace['test_predict'] = matlab.double( test_predict.tolist() ) 

    # within MATLAB, call this to plot
    # >>plot( [testY, test_predict], '+-' )