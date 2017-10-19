import input_data
import tensorflow as tf

# create logical refgression model
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_lr = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y*tf.log(y_lr))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_lr, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# train logical refgression model
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for i in range(1000):
  	batch_x, batch_y = mnist.train.next_batch(100)
  	sess.run(train_step, feed_dict = {x: batch_x, y: batch_y})

print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels}))
