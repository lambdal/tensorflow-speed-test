import tensorflow as tf


def input_fn(x, y, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat(1000)
  dataset = dataset.apply(
      tf.contrib.data.batch_and_drop_remainder(batch_size))
  dataset = dataset.prefetch(2)
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()


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
