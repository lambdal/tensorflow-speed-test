"""
Reference performance on 1080 TI
1 GPU:
2 GPU:
3 GPU:
4 GPU:

"""
from __future__ import print_function
import time
import argparse
import sys


import tensorflow as tf

import net


ps_ops = ["Variable", "VariableV2", "AutoReloadVariable"]


def assign_to_device(device, ps_device="/cpu:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            return "/" + ps_device
        else:
            return device
    return _assign


def batch_split(batch, idx, batch_size_per_gpu):
  bs_per_gpu = batch_size_per_gpu
  batch_per_gpu = ()
  for x in batch:
    batch_per_gpu = (batch_per_gpu +
                     (x[idx * bs_per_gpu:(idx + 1) * bs_per_gpu],))
  return batch_per_gpu


def average_gradients(tower_grads):
  average_grads = []

  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      if g is not None:
        expanded_g = tf.expand_dims(g, 0)
        grads.append(expanded_g)

    if grads:
      # Average over the "tower" dimension.
      grad = tf.concat(grads, 0)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So we will just return the first tower"s pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
  return average_grads


def print_trainable_variables():

  for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print (i)

  print(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))


def print_global_variables():

  for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print (i.name)

  print(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))


def main():

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--list_devices",
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
  config = parser.parse_args()

  config.list_devices = list(map(int, config.list_devices.split(",")))
  config.gpu_count = len(config.list_devices)

  with tf.device("/cpu:0"):

    list_grads_and_vars = []

    # Map
    for split_id, device_id in enumerate(config.list_devices):
      with tf.device(assign_to_device("/gpu:{}".format(device_id),
                     ps_device="/cpu:0")):

        images_batch = tf.constant(1.0,
                                   shape=[config.batch_size_per_gpu, 224, 224, 3],
                                   dtype=tf.float32)
        labels_batch = tf.constant(1,
                                   shape=[config.batch_size_per_gpu],
                                   dtype=tf.int32)

        outputs = net.simple_net(images_batch,
                                 config.batch_size_per_gpu,
                                 config.num_classes)

        loss = tf.losses.sparse_softmax_cross_entropy(
          labels=labels_batch, logits=outputs)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.001,
            momentum=0.9)

        list_grads_and_vars.append(optimizer.compute_gradients(loss))

    ave_grads_and_vars = average_gradients(list_grads_and_vars)

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
