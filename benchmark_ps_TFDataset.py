"""
Reference performance on 1080 TI
1 GPU:
2 GPU:
3 GPU:
4 GPU:


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
import utils


def main():

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--device_list",
                      help="Comma seperatted device IDs to run benchmark on.",
                      type=str,
                      default="0")
  parser.add_argument("--batch_size_per_gpu",
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
  parser.add_argument("--num_samples",
                      help="Number of samples in the dataset.",
                      type=int,
                      default=128)
  config = parser.parse_args()

  config.device_list = list(map(int, config.device_list.split(",")))
  config.gpu_count = len(config.device_list)

  x = np.random.rand(
    config.gpu_count * config.batch_size_per_gpu, 224, 224, 3).astype("f")
  y = np.random.randint(config.num_classes,
                        size=(config.gpu_count * config.batch_size_per_gpu))

  with tf.device("/cpu:0"):
    batch = net.input_fn(x, y, config.gpu_count * config.batch_size_per_gpu)

  list_grads_and_vars = []

  # Map
  for split_id, device_id in enumerate(config.device_list):
    with tf.device(utils.assign_to_device("/gpu:{}".format(device_id),
                   ps_device="/cpu:0")):

      # Split input data across multiple devices
      images_batch, lables_batch = utils.batch_split(batch,
                                                     split_id,
                                                     config.batch_size_per_gpu)

      outputs = net.simple_net(images_batch,
                               config.batch_size_per_gpu,
                               config.num_classes)

      loss = tf.losses.sparse_softmax_cross_entropy(
        labels=lables_batch, logits=outputs)

      optimizer = tf.train.MomentumOptimizer(
          learning_rate=0.001,
          momentum=0.9)

      list_grads_and_vars.append(optimizer.compute_gradients(loss))

  ave_grads_and_vars = utils.average_gradients(list_grads_and_vars)

  minimize_op = optimizer.apply_gradients(ave_grads_and_vars)

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
      sess.run(minimize_op)
    print("Warm up finished.")

    start_time = time.time()
    for i_iter in range(config.num_iterations):
      print("\rIteration: " + str(i_iter), end="")
      sys.stdout.flush()
      sess.run(minimize_op)
    end_time = time.time()

    total_time = end_time - start_time
    total_num_images = (
      config.gpu_count * config.batch_size_per_gpu * config.num_iterations)

    print("\nTotal time spend: " + str(total_time) + " secs.")
    print("Average Speed: " +
          str(total_num_images / total_time) +
          " images/sec.")


if __name__ == "__main__":
  main()
