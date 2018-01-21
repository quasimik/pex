from flask import Flask
app = Flask(__name__)

import tensorflow as tf
import sys

@app.route('/')
def index():
	return 'hey'

@app.route('/tfroute')
def tfroute():
	W = tf.Variable([0.3], tf.float32)
	b = tf.Variable([-0.3], tf.float32)
	x = tf.placeholder(tf.float32)
	linear_model = W*x + b

	y = tf.placeholder(tf.float32)
	loss = tf.reduce_sum(tf.square(linear_model - y))

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	# fixW = tf.assign(W, [-1.])
	# fixb = tf.assign(b, [1.])
	# sess.run([fixW, fixb])

	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss)

	# Training data
	x_train = [1, 2, 3, 4] # Input data
	y_train = [0, -1, -2, -3] # Expected output

	for i in range(1000):
		sess.run(train, {x: x_train, y: y_train})

	print(sess.run([W, b, loss], {x: x_train, y: y_train}))
	return('ok')