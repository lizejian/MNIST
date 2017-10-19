# coding:utf8

import tensorflow as tf
import input_data

def initialize_weight(shape, stddev, name):
	initial = tf.truncated_normal(shape, dtype = tf.float32, stddev = stddev)
	return tf.Variable(initial, name = name)

def initialize_bias(shape):
	initial = tf.random_normal(shape)
	return tf.Variable(initial)

def conv2d(x, w, b):
	return tf.nn.relu((tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME') + b))

def max_pool(x, f):
	return tf.nn.max_pool(x, ksize = [1, f, f, 1], strides = [1, 1, 1, 1], padding = 'SAME')

# Create AlexNet Model
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, shape = [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# layer1: conv2d
w1_conv = initialize_weight([3, 3, 1, 64], 0.1, 'w1')
b1_conv = initialize_bias([64])
h1_conv = conv2d(x_image, w1_conv, b1_conv)#28x28x1>>28x28x64
h1_pool = max_pool(h1_conv, 2)#28x28x64>>14x14x64

# layer2: conv2d
w2_conv = initialize_weight([3, 3, 64, 64], 0.1, 'w2')
b2_conv = initialize_bias([64])
h2_conv = conv2d(h1_pool, w2_conv, b2_conv)#14x14x64>>14x14x64
h2_pool = max_pool(h2_conv, 2)#14x14x64>>7x7x64

# layer3: conv2d
w3_conv = initialize_weight([3, 3, 64, 128], 0.1, 'w3')
b3_conv = initialize_bias([128])
h3_conv = conv2d(h2_pool, w3_conv, b3_conv)#7x7x64>>7x7x128

# layer4: conv2d
w4_conv = initialize_weight([3, 3, 128, 128], 0.1, 'w4')
b4_conv = initialize_bias([128])
h4_conv = conv2d(h3_conv, w4_conv, b4_conv)#7x7x128>>7x7x128

# layer5: conv2d
w5_conv = initialize_weight([3, 3, 128, 256], 0.1, 'w5')
b5_conv = initialize_bias([256])
h5_conv = conv2d(h4_conv, w5_conv, b5_conv)#7x7x128>>7x7x256
h5_pool = max_pool(h5_conv, 2)#
shape = h5_pool.get_shape() 
h5_pool_flat = tf.reshape(h5_pool, [-1, shape[1].value*shape[2].value*shape[3].value])

# layer6: full connection
w6_fc = initialize_weight([256*28*28, 1024], 0.01, 'w6')
b6_fc = initialize_bias([1024])
h6_fc = tf.nn.relu(tf.matmul(h5_pool_flat, w6_fc) + b6_fc)
keep_prob = tf.placeholder('float')
h6_drop = tf.nn.dropout(h6_fc, keep_prob = keep_prob)

# layer7: full connection
w7_fc = initialize_weight([1024, 1024], 0.01, 'w7')
b7_fc = initialize_bias([1024])
h7_fc = tf.nn.relu(tf.matmul(h6_drop, w7_fc) + b7_fc)
h7_drop = tf.nn.dropout(h7_fc, keep_prob = keep_prob)

# layer8: softmax
w8_sf = initialize_weight([1024, 10], 0.01, 'w8')
b8_sf = initialize_bias([10])
y_conv = tf.nn.softmax(tf.matmul(h7_drop, w8_sf) + b8_sf)

cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train AlexNet Model
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1500):
	batch = mnist.train.next_batch(64)
	if i % 10 == 0:
		train_accuracy = sess.run(accuracy, feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.9})
		print('step %d, training accuracy %g' % (i, train_accuracy))
	sess.run(train_step, feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.9})

test_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 0.9})
print('test accuracy %g' % test_accuracy)
sess.close()