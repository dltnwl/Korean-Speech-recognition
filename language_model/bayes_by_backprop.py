from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

# Dependency imports

import numpy as np
import six
import sonnet as snt
from sonnet.examples import ptb_reader
import sonnet.python.custom_getters.bayes_by_backprop as bbb
import tensorflow as tf
from glob import glob
import re as re
import os as os
import json as _json
import os.path as path
import utils as utils
import random as _random
import utils.hangul as _hangul


from importlib import import_module as import_module
from functools import wraps

from six.moves import xrange as range

from abc import ABC, abstractmethod

import pandas as pd
import random as random
import tensorflow as tf
from tqdm import tqdm

from sonnet.python.modules import base
import pickle as _pickle
import os.path as _path
from random import shuffle as _shuffle
import numpy as _np

learning_rate=0.0001
n_classes=56

batch_size=40
feat_len =13
hidden_size=512




_encode_map = {c: i for i, c in enumerate(labels)}
_decode_map = labels + [""]


def encode(texts):
  return list(map(lambda s: "".join(map(lambda c: _encode_map[c], s)), sequences))



def decode(sequences):
  return list(map(lambda s: "".join(map(lambda c: _decode_map[c], s)), sequences))


nest = tf.contrib.framework.nest


batch_size=20
n_layers= 2
num_training_epochs= 1

logdir='/home/suji93/sonnet/sonnet/examples/log/acoustic22'
prior_pi= 0.25
prior_sigma1= np.exp(-1.0)
prior_sigma2= np.exp(-7.0)
print_every_batches= 500




tf.logging.set_verbosity(tf.logging.INFO)


_LOADED = {}
DataOps = collections.namedtuple("DataOps", "sparse_obs sparse_target")


def _run_session_with_no_hooks(sess, *args, **kwargs):
  """Only runs of the training op should contribute to speed measurement."""
  return sess._tf_sess().run(*args, **kwargs)  # pylint: disable=protected-access



class GlobalNormClippingOptimizer(tf.train.Optimizer):
  """Optimizer that clips gradients by global norm."""

  def __init__(self,
               opt,
               clip_norm,
               use_locking=False,
               name="GlobalNormClippingOptimizer"):
    super(GlobalNormClippingOptimizer, self).__init__(use_locking, name)

    self._opt = opt
    self._clip_norm = clip_norm

  def compute_gradients(self, *args, **kwargs):
    return self._opt.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, *args, **kwargs):
    if self._clip_norm == np.inf:
      return self._opt.apply_gradients(grads_and_vars, *args, **kwargs)
    grads, vars_ = zip(*grads_and_vars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, self._clip_norm)
    return self._opt.apply_gradients(zip(clipped_grads, vars_), *args, **kwargs)


class CustomScaleMixture(object):

  def __init__(self, pi, sigma1, sigma2):
    self.mu, self.pi, self.sigma1, self.sigma2 = map(
        np.float32, (0.0, pi, sigma1, sigma2))

  def log_prob(self, x):
    n1 = tf.contrib.distributions.Normal(self.mu, self.sigma1)
    n2 = tf.contrib.distributions.Normal(self.mu, self.sigma2)
    mix1 = tf.reduce_sum(n1.log_prob(x), -1) + tf.log(self.pi)
    mix2 = tf.reduce_sum(n2.log_prob(x), -1) + tf.log(np.float32(1.0 - self.pi))
    prior_mix = tf.stack([mix1, mix2])
    lse_mix = tf.reduce_logsumexp(prior_mix, [0])
    return tf.reduce_sum(lse_mix)


def custom_scale_mixture_prior_builder(getter, name, *args, **kwargs):
  """A builder for the gaussian scale-mixture prior of Fortunato et al.

  Please see https://arxiv.org/abs/1704.02798, section 7.1

  Args:
    getter: The `getter` passed to a `custom_getter`. Please see the
      documentation for `tf.get_variable`.
    name: The `name` argument passed to `tf.get_variable`.
    *args: Positional arguments forwarded by `tf.get_variable`.
    **kwargs: Keyword arguments forwarded by `tf.get_variable`.

  Returns:
    An instance of `tf.contrib.distributions.Distribution` representing the
    prior distribution over the variable in question.
  """
  # This specific prior formulation doesn't need any of the arguments forwarded
  # from `get_variable`.
  del getter
  del name
  del args
  del kwargs
  return CustomScaleMixture(
      prior_pi, prior_sigma1, prior_sigma2)


def lstm_posterior_builder(getter, name, *args, **kwargs):
  """A builder for a particular diagonal gaussian posterior.

  Args:
    getter: The `getter` passed to a `custom_getter`. Please see the
      documentation for `tf.get_variable`.
    name: The `name` argument passed to `tf.get_variable`.
    *args: Positional arguments forwarded by `tf.get_variable`.
    **kwargs: Keyword arguments forwarded by `tf.get_variable`.

  Returns:
    An instance of `tf.contrib.distributions.Distribution` representing the
    posterior distribution over the variable in question.
  """
  del args
  parameter_shapes = tf.contrib.distributions.Normal.param_static_shapes(
      kwargs["shape"])

  # The standard deviation of the scale mixture prior.
  prior_stddev = np.sqrt(
      prior_pi * np.square(prior_sigma1) +
      (1 - prior_pi) * np.square(prior_sigma2))

  loc_var = getter(
      "{}/posterior_loc".format(name),
      shape=parameter_shapes["loc"],
      initializer=kwargs.get("initializer"),
      dtype=tf.float32)
  scale_var = getter(
      "{}/posterior_scale".format(name),
      initializer=tf.random_uniform(
          minval=np.log(np.exp(prior_stddev / 4.0) - 1.0),
          maxval=np.log(np.exp(prior_stddev / 2.0) - 1.0),
          dtype=tf.float32,
          shape=parameter_shapes["scale"]))
  return tf.contrib.distributions.Normal(
      loc=loc_var,
      scale=tf.nn.softplus(scale_var) + 1e-5,
      name="{}/posterior_dist".format(name))


def non_lstm_posterior_builder(getter, name, *args, **kwargs):

  del args
  parameter_shapes = tf.contrib.distributions.Normal.param_static_shapes(
      kwargs["shape"])

  # The standard deviation of the scale mixture prior.
  prior_stddev = np.sqrt(
      prior_pi * np.square(prior_sigma1) +
      (1 - prior_pi) * np.square(prior_sigma2))

  loc_var = getter(
      "{}/posterior_loc".format(name),
      shape=parameter_shapes["loc"],
      initializer=kwargs.get("initializer"),
      dtype=tf.float32)
  scale_var = getter(
      "{}/posterior_scale".format(name),
      initializer=tf.random_uniform(
          minval=np.log(np.exp(prior_stddev / 2.0) - 1.0),
          maxval=np.log(np.exp(prior_stddev / 1.0) - 1.0),
          dtype=tf.float32,
          shape=parameter_shapes["scale"]))
  return tf.contrib.distributions.Normal(
      loc=loc_var,
      scale=tf.nn.softplus(scale_var) + 1e-5,
      name="{}/posterior_dist".format(name))





estimator_mode = tf.constant(bbb.EstimatorModes.sample)

lstm_bbb_custom_getter = bbb.bayes_by_backprop_getter(
    posterior_builder=lstm_posterior_builder,
    prior_builder=custom_scale_mixture_prior_builder,
    kl_builder=bbb.stochastic_kl_builder,
    sampling_mode_tensor=estimator_mode)

non_lstm_bbb_custom_getter = bbb.bayes_by_backprop_getter(
    posterior_builder=non_lstm_posterior_builder,
    prior_builder=custom_scale_mixture_prior_builder,
    kl_builder=bbb.stochastic_kl_builder,
    sampling_mode_tensor=estimator_mode)



class Linear(base.AbstractModule):
  """Linear module, optionally including bias."""

  def __init__(self,
               output_size,
               use_bias=True,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="linear"):

    super(Linear, self).__init__(custom_getter=custom_getter, name=name)
    self._output_size = output_size
    self._use_bias = use_bias
    self._input_shape = None
    self._w = None
    self._b = None

    n_classes=self._output_size
  def _build(self, inputs):


    self._w = tf.get_variable("w",
                              shape=[2*512, n_classes],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    outputs = tf.matmul(inputs, self._w)


    self._b = tf.get_variable("b",
                                shape=[n_classes],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer(dtype=tf.float32))
    outputs += self._b
    logits=tf.transpose(tf.reshape(outputs, [batch_size, -1,n_classes]), (1,0, 2))

    return logits





def build_module(inputs, input_len):
    cores = [snt.LSTM(hidden_size,
                      custom_getter=lstm_bbb_custom_getter,
                      forget_bias=0.0,
                      name="lstm_layer_{}".format(i))
             for i in six.moves.range(n_layers)]
    
    rnn_core = snt.DeepRNN(cores,skip_connections=True,name="deep_lstm_core")
    
    # Do BBB on weights but not biases of output layer.
    output_linear = Linear(n_classes, custom_getter={"w": non_lstm_bbb_custom_getter})


    
#    initial_rnn_state = nest.map_structure(lambda t: tf.get_local_variable( 
#            "{}/rnn_state/{}".format("train", t.op.name), initializer=t), rnn_core.initial_state(batch_size))
#    assign_zero_rnn_state = nest.map_structure(lambda x: x.assign(tf.zeros_like(x)), initial_rnn_state)
#    assign_zero_rnn_state = tf.group(*nest.flatten(assign_zero_rnn_state))
    
    # Unroll the RNN core over the sequence.
    rnn_output_seq, rnn_final_state = tf.nn.dynamic_rnn(
        cell=rnn_core,
        inputs=inputs,
        sequence_length=input_len,
         dtype=tf.float32)
    
    # Persist the RNN state for the next unroll.
#    update_rnn_state = nest.map_structure(tf.assign, initial_rnn_state, rnn_final_state)
#    with tf.control_dependencies(nest.flatten(update_rnn_state)):
#      rnn_output_seq = tf.identity(rnn_output_seq, name="rnn_output_seq")
    
    rnn_output_seq=tf.reshape(tf.concat(rnn_output_seq, 2), [-1, 2*512]) 
    output_logits = output_linear(rnn_output_seq)
    return output_logits#, assign_zero_rnn_state


with tf.device("/gpu:0"):
    optimizer = GlobalNormClippingOptimizer(tf.train.GradientDescentOptimizer(learning_rate=0.001), clip_norm=5.0)
    
    inputs = tf.placeholder(tf.float32, [None, None,feat_len],name='inputs')
    input_len = tf.placeholder(tf.int32, [None],name='input_len')
    output=tf.sparse_placeholder(tf.int32,name='output')
    #output= tf.placeholder(tf.int32, [None],name='output')
    
logit=build_module(inputs, input_len)


data_loss= tf.reduce_mean(tf.nn.ctc_loss(output,logit,input_len,ignore_longer_outputs_than_inputs=True))
        
total_kl_cost = bbb.get_total_kl_cost()
    
scaled_kl_cost = total_kl_cost / batch_size
total_loss = tf.add(scaled_kl_cost, data_loss)
tf.summary.scalar("LOSS", total_loss)
summary=tf.summary.merge_all()

global_step = tf.get_variable("num_weight_updates",
  initializer=tf.constant(0, dtype=tf.int32, shape=()),
  collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

# Optimize as usual.

with tf.control_dependencies([optimizer.minimize(total_loss)]):
    global_step_and_train = global_step.assign_add(1)



predictions = tf.cast(tf.argmax(logit, axis=-1), tf.int32, name="pred")
decoded,_=tf.nn.ctc_beam_search_decoder(logit, input_len, beam_width=100)
ler=tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), output))
decoded=tf.sparse_tensor_to_dense(decoded[0],-1)
label_probs = tf.nn.softmax(logit, dim=-1)

        
def _run_session_with_no_hooks(sess, *args, **kwargs):
  """Only runs of the training op should contribute to speed measurement."""
  return sess._tf_sess().run(*args, **kwargs)  # pylint: disable=protected-access
        
        
epoch=30
print_every_batches=10

summary_writer = tf.summary.FileWriterCache.get(logdir=logdir)
tf.logging.info("Beginning training for {} epochs, each with {} batches.".format(epoch, batch_size))
with tf.train.MonitoredTrainingSession(is_chief=True, checkpoint_dir=logdir, save_summaries_secs=10) as sess:
  num_updates_v = _run_session_with_no_hooks(sess, global_step)
  valid_cost=0
  for epoch_idx in range(20):
    epoch_ler=0
    epoch_cost = 0
    for _ in range(int(n_train/batch_size)):
        
      summ,train_cost, num_updates_v, train_ler = sess.run([summary,total_loss, global_step_and_train, ler], feed_dict=feed(list(next(batchs["train"]))))
      epoch_cost += train_cost
      epoch_ler+=train_ler
      summary_writer.add_summary(summ,num_updates_v)
    valid_cost_v, num_updates_v, valid_ler = sess.run([total_loss, global_step, ler], feed_dict=feed(list(next(batchs["valid"]))))
    # Run a training epoch.

    print('train loss:',epoch_cost/n_train)
    print('train ler:', epoch_ler/n_train)
    print('valid loss:', valid_cost_v)
    print('vliad ler:', valid_ler)
    

    
    
    
