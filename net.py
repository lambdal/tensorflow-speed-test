import tensorflow as tf


def simple_net(image, batch_size, num_classes):
  with tf.variable_scope("simple_net", reuse=tf.AUTO_REUSE):
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
    output = tf.reshape(output, [batch_size, -1])
    output = tf.layers.dense(output, num_classes)

  return output
