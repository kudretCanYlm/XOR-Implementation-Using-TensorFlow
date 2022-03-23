import tensorflow as tf
import tensorboard

import numpy as np

# XOR Implementation Using TensorFlow

# XOR implementation in Tensorflow with hidden layers being sigmoid to
# introduce Non-Linearity
# Create placeholders for training input and output labels
tf.compat.v1.disable_v2_behavior()

x_ = tf.compat.v1.placeholder(tf.float32, shape=[4, 2], name="x-input")
y_ = tf.compat.v1.placeholder(tf.float32, shape=[4, 1], name="y-input")

# Define the weights to the hidden and output layer respectively.
w1 = tf.Variable(tf.compat.v1.random_uniform([2, 2], -1, 1), name="Weights1")
w2 = tf.Variable(tf.compat.v1.random_uniform([2, 1], -1, 1), name="Weights2")

# Define the bias to the hidden and output layers respectively

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Define the final output through forward pass
z2 = tf.sigmoid(tf.matmul(x_, w1)+b1)
pred = tf.sigmoid(tf.matmul(z2, w2)+b2)

# Define the Cross-entropy/Log-loss Cost function based on the output label y and
# the predicted probability by the forward pass
cost = tf.reduce_mean(((y_*tf.compat.v1.log(pred)) +
                      ((1 - y_) * tf.compat.v1.log(1.0 - pred))) * -1)
learning_rate = 0.001

train_step = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate).minimize(cost)

# Now that we have all that we need set up we will start the training
XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [[0], [1], [1], [0]]

# Initialize the variables
init = tf.compat.v1.initialize_all_variables()
sess = tf.compat.v1.Session()
writer = tf.compat.v1.summary.FileWriter(
    "./Downloads/XOR_logs", sess.graph_def)

sess.run(init)

for i in range(20):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})

print('Final Prediction', sess.run(pred, feed_dict={x_: XOR_X, y_: XOR_Y}))

