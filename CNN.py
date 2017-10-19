import input_data
import tensorflow as tf


def initialize_weight(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def initialize_bias(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


#create CNN Model
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])#28x28x1

# layer1:conv2d
W1_conv = initialize_weight([5, 5, 1, 32])
b1_conv = initialize_bias([32])
h1_conv = tf.nn.relu(conv2d(x_image, W1_conv) + b1_conv)#28x28x1>>28x28x32
h1_pool = max_pool_2x2(h1_conv)#28x28x32>>14x14x32

# layer2:conv2d
W2_conv = initialize_weight([5, 5, 32, 64])
b2_conv = initialize_bias([64])
h2_conv = tf.nn.relu(conv2d(h1_pool, W2_conv) + b2_conv)#14x14x32>>14x14x64
h2_pool = max_pool_2x2(h2_conv)#14x14x64>>7x7x64
h2_pool_flat = tf.reshape(h2_pool, [-1, 7*7*64])#7x7x64>>7*7*64

# layer3:full connection
W3_fc = initialize_weight([7*7*64, 1024])
b3_fc = initialize_bias([1024])
h3_fc = tf.nn.relu(tf.matmul(h2_pool_flat, W3_fc) + b3_fc)#7*7*64>>1024

# drop out
keep_prob = tf.placeholder('float')
h3_fc_drop = tf.nn.dropout(h3_fc, keep_prob)

# layer4: full connection
W4_fc = initialize_weight([1024, 10])
b4_fc = initialize_bias([10])
y_conv = tf.nn.softmax(tf.matmul(h3_fc_drop, W4_fc) + b4_fc)

cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# train model
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i % 100 == 0:
		train_accuracy = sess.run(accuracy, feed_dict = {x: batch[0], y: batch[1], keep_prob: 1.0})
		print('step %d, training accuracy %g' % (i, train_accuracy))
	sess.run(train_step, feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5})

test_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
print('test accuracy %g' % test_accuracy)
sess.close()

