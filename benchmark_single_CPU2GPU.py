"""
Reference performance on 1080 TI

"""
from __future__ import print_function
import time
import numpy as np
import argparse
import sys
# By default CUDA guesses which device is fastest using a simple heuristic,
# and make that device 0. This is difficult to debug. We use PCI_BUS_ID
# instead to orders devices by PCI bus ID in ascending order.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import tensorflow as tf

import net


def main():

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--device_id",
                      help="Comma seperatted device IDs to run benchmark on.",
                      type=int,
                      default=0)
  parser.add_argument("--batch_size",
                      help="Batch size on each GPU",
                      type=int,
                      default=32)
  parser.add_argument("--num_classes",
                      help="Number of classes",
                      type=int,
                      default=100)
  parser.add_argument("--num_warmup",
                      help="Number of warm up iterations.",
                      type=int,
                      default=50)
  parser.add_argument("--num_iterations",
                      help="Number of benchmark iterations.",
                      type=int,
                      default=200)
  config = parser.parse_args()

  x = np.random.rand(config.batch_size, 224, 224, 3)
  y = np.random.randint(config.num_classes, size=(config.batch_size))

  with tf.device("/gpu:{}".format(config.device_id)):
    image = tf.placeholder(
      tf.float32, shape=(config.batch_size, 224, 224, 3))
    label = tf.placeholder(
      tf.int32, shape=(config.batch_size))

    outputs = net.simple_net(image, config.batch_size, config.num_classes)

    loss = tf.losses.sparse_softmax_cross_entropy(
      labels=label, logits=outputs)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=0.001,
        momentum=0.9)

    grads_and_vars = optimizer.compute_gradients(loss)
    minimize_op = optimizer.apply_gradients(
      grads_and_vars)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90,
                              allow_growth=True)

  session_config = tf.ConfigProto(
    allow_soft_placement=False,
    log_device_placement=False,
    gpu_options=gpu_options)

  with tf.Session(config=session_config) as sess:
    sess.run(tf.global_variables_initializer())

    print("Warm up started.")
    for i_iter in range(config.num_warmup):
      sess.run(minimize_op, feed_dict={image: x, label: y})
      # sess.run([image, label], feed_dict={image: x, label: y})
    print("Warm up finished.")

    start_time = time.time()
    for i_iter in range(config.num_iterations):
      print("\rIteration: " + str(i_iter), end="")
      sess.run(minimize_op, feed_dict={image: x, label: y})
      # sess.run([image, label], feed_dict={image: x, label: y})
      sys.stdout.flush()
    end_time = time.time()

    total_time = end_time - start_time
    total_num_images = config.batch_size * config.num_iterations

    print("\nTotal time spend: " + str(total_time) + " secs.")
    print("Average Speed: " +
          str(total_num_images / total_time) +
          " images/sec.")


if __name__ == "__main__":
  main()
