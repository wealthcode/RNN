'''
This script shows how to predict stock prices using a basic RNN
'''

import tensorflow as tf
import numpy as np
import matplotlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.set_random_seed(777)  # reproducibility

#if "DISPLAY" not in os.environ:
	# remove Travis CI Error
#	matplotlib.use('Agg')

import matplotlib.pyplot as plt


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


# train Parameters
seq_length = 30
data_dim = 9
hidden_dim = 20
output_dim = 3
learning_rate = 0.01
iterations = 5000

# Open, High, Low, Volume, Close
xy = np.loadtxt('_sim_flag3_R_P_Y_Cpp.dat')
#xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy[:, 1:]
y = xy[:, -3:]  # Close as label

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
	_x = x[i:i + seq_length]
	_y = y[i + seq_length]  # Next close price
	#print(_x, "->", _y)
	dataX.append(_x)
	dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
	dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
	dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 3])

# build a LSTM network
#cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    return cell

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
	outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 3])
predictions = tf.placeholder(tf.float32, [None, 3])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	# Training step
	for i in range(iterations):
		_, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
		if step_loss < 1:
			break
		print("[step: {}] loss: {}".format(i, step_loss))

	# Test step
	test_predict = sess.run(Y_pred, feed_dict={X: testX})
	rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
	print("RMSE: {}".format(rmse_val))

	# Plot predictions
	plt.plot(testY[:, 0], label='Roll_test')
	plt.plot(test_predict[:, 0], label='Roll_predicted')
	plt.plot(testY[:, 1], label='Pitch_test')
	plt.plot(test_predict[:, 1], label='Pitch_predicted')
	plt.plot(testY[:, 2], label='Yaw_test')
	plt.plot(test_predict[:, 2], label='Yaw_predicted')
	plt.xlabel("Time Period")
	plt.ylabel("Angle [rad]")
	plt.legend(loc='lower right')
	plt.show()