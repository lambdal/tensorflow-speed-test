"""
Reference performance on Titan X
Average Speed: 369.520257291 images/sec.

"""
import time
import numpy as np

import tensorflow as tf


BATCH_SIZE = 32
NUM_CLASSES = 1000

NUM_WARMUP = 50
NUM_ITER = 200

x = np.random.rand(BATCH_SIZE, 224, 224, 3)
y = np.random.randint(NUM_CLASSES, size=(BATCH_SIZE))

image = tf.placeholder(
  tf.float32, shape=(BATCH_SIZE, 224, 224, 3))
label = tf.placeholder(
  tf.int32, shape=(BATCH_SIZE))


def net(image):
  output = tf.layers.conv2d(image, 64, 3)
  output = tf.layers.max_pooling2d(output, 2, 2)
  output = tf.layers.conv2d(output, 64, 3)
  output = tf.layers.max_pooling2d(output, 2, 2)
  output = tf.layers.conv2d(output, 64, 3)
  output = tf.layers.max_pooling2d(output, 2, 2)
  output = tf.layers.conv2d(output, 64, 3)
  output = tf.layers.max_pooling2d(output, 2, 2)
  output = tf.layers.conv2d(output, 64, 3)
  output = tf.layers.max_pooling2d(output, 2, 2)
  output = tf.reshape(output, [BATCH_SIZE, -1])
  output = tf.layers.dense(output, NUM_CLASSES)
  return output


output = net(image)

loss = tf.losses.softmax_cross_entropy(
  logits=output, onehot_labels=tf.one_hot(label, NUM_CLASSES))

optimizer = tf.train.MomentumOptimizer(
    learning_rate=0.001,
    momentum=0.9)

grads_and_vars = optimizer.compute_gradients(loss)
minimize_op = optimizer.apply_gradients(
  grads_and_vars)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  print("Warm up finished.")
  for i_iter in range(NUM_WARMUP):
    _loss = sess.run(minimize_op, feed_dict={image: x, label: y})
  print("Warm up finished.")

  start_time = time.time()
  for i_iter in range(NUM_ITER):
    print(i_iter)
    _loss = sess.run(minimize_op, feed_dict={image: x, label: y})
  end_time = time.time()

  total_time = end_time - start_time
  total_num_images = BATCH_SIZE * NUM_ITER
  print("Average Speed: " +
        str(total_num_images / total_time) +
        " images/sec.")
