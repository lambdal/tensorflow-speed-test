import tensorflow as tf


ps_ops = ["Variable", "VariableV2", "AutoReloadVariable"]


def assign_to_device(device, ps_device="/cpu:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            return ps_device
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
