from glob import glob
import re as re
import os as os
import json as _json
import os.path as path
import soundfile as sf
import utils as utils
import random as _random
import utils.hangul as _hangul

from random import shuffle as shuffle
from importlib import import_module as import_module
from functools import wraps
from python_speech_features import mfcc
import time
import scipy.io.wavfile as wav
import numpy as np
from six.moves import xrange as range

from abc import ABC, abstractmethod

import pandas as pd
import random as random
import tensorflow as tf
from tqdm import tqdm


import pickle as _pickle
import os.path as _path
from random import shuffle as _shuffle
from importlib import import_module as _import_module
import numpy as _np

params = tf.contrib.training.HParams(
    learning_rate=0.002,
    n_classes=54,
    train_steps=10,
    min_eval_frequency=100,
    use_fp16=False,
    num_filters=5,
    temporal_stride=1,
    keep_prob=0.5,
    batch_size=100,
    num_hidden=50,
    num_rnn_layers=3,
    rnn_type='uni-dir'
)

NUM_CLASSES =54
feat_len = 13
n_features=50

ctc_labels = _hangul.JAMOS
labels = [" "] + ctc_labels

def load_data(data_dir,n_features):

  assert _path.isdir(data_dir)
  load_rawdata = getattr(_import_module("utils.data"),"nangdok")
  preprocess_audio = getattr(_import_module("utils.audio"), "mfcc")
  preprocess_text = getattr(_import_module("utils.hangul"), "jamo_token")
  cache_path = _path.join(data_dir, "nangdok" + "_cache.pkl")

  print("Loading cached data.")
  with open(cache_path, "rb") as f:
      cache = _pickle.load(f)
  return [(audio, preprocess_text(text)) for (audio, text) in cache]


data=load_data('/home/suji93/data/',13)
data = list(filter(lambda x: x[0] is not None,data))



_encode_map = {c: i for i, c in enumerate(labels)}
_decode_map = labels + [""]
def encode(texts):
  indices = []
  values = []
  for index, text in enumerate(texts):
    indices.extend(zip([index] * len(text), range(len(text))))
    values.extend(map(lambda c: _encode_map[c], text))
  indices = np.asarray(indices, dtype=np.int64)
  values = np.asarray(values, dtype=np.int32)
  shape = np.asarray([len(texts), np.asarray(indices).max(0)[1] + 1],dtype=np.int64)
  return indices, values, shape

def decode(sequences):
  """Decode CTC output to texts."""
  return list(map(lambda s: "".join(map(lambda c: _decode_map[c], s)), sequences))




def acoustic_model(feats, seq_len,targets, params, is_train):
    with tf.variable_scope('conv1') as scope:

        
        output_dropout_prob=0.3
        state_dropout_prob=0.3
        output_keep_prob = tf.cond(is_train, lambda: 1.0-output_dropout_prob, lambda: 1.0)
        state_keep_prob = tf.cond(is_train, lambda: 1.0-state_dropout_prob, lambda: 1.0)
        
        kernel = tf.get_variable('weights',
            shape=[10, 1, 1, params.num_filters],initializer=\
            tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                       mode='FAN_IN',
                                                       uniform=False,
                                                       seed=None,
                                                       dtype= tf.float32), dtype=tf.float32)

        feats = tf.expand_dims(feats, dim=-1)
        conv = tf.nn.conv2d(feats, kernel,
                            [1, params.temporal_stride, 1, 1],padding='SAME')
        # conv = tf.nn.atrous_conv2d(feats, kernel, rate=2, padding='SAME')
        biases = tf.get_variable('biases', [params.num_filters],
                                  initializer=tf.constant_initializer(0),
                                  dtype=tf.float32)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        conv1_drop = tf.nn.dropout(conv1, output_keep_prob)

    # recurrent layers
    with tf.variable_scope('rnn') as scope:
        
        rnn_input = tf.reshape(conv1_drop, [params.batch_size, -1,feat_len*params.num_filters])

        
        
        cells = [tf.contrib.rnn.LSTMCell(100) for _ in range(2)]
        cells = [tf.contrib.rnn.DropoutWrapper(cell,
                                           output_keep_prob=output_keep_prob,
                                           state_keep_prob=state_keep_prob,
                                           variational_recurrent=True,
                                           dtype=tf.float32)
             for cell in cells]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(*cells, rnn_input, seq_len, dtype=tf.float32)
        outputs = tf.reshape(tf.concat(outputs, 2), [-1, 2 * 100])
        
        
        # Fully connected layer
        W = tf.Variable(tf.truncated_normal([2 * 100, params.n_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[params.n_classes]))
        logits = tf.transpose(tf.reshape(tf.matmul(outputs, W) + b, [tf.shape(feats)[0], -1, params.n_classes]), (1, 0, 2))     

        decoded,_=tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=100)
        with tf.name_scope("Loss"):
            ctc = tf.reduce_mean(tf.nn.ctc_loss(targets,logits,seq_len,ignore_longer_outputs_than_inputs=True))
        
        with tf.name_scope("ler"):
            ler=tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
        with tf.name_scope("Decoded"):
            decoded=tf.sparse_tensor_to_dense(decoded[0],-1)
        


        return ctc,decoded, ler

def feed(data, istrain):
    text=encode(data[1])

    input_len_feed = np.asarray(list(map(len, data[0])))
    inputs_feed = np.zeros((len(data[0]),
                             max(input_len_feed),
                             feat_len),
                            np.float32)
    for i, l in enumerate(input_len_feed):
      inputs_feed[i, :l, :] = data[0][i]
    return {inputs: inputs_feed,
            input_len: input_len_feed,
            output: text,
            is_train:istrain}


def batch_generator(data, batch_size,idx=0, num_workers=1, bin_size=100, bin_num=10):

  tmp = []
  bin_idx = 0
  for _ in range(bin_num):
    tmp.append(list(filter(lambda x: bin_idx <= len(x) and len(x) < bin_idx+bin_size,data)))
    bin_idx += bin_size
  order = list(range(0, len(data), batch_size))
  while True:
    shuffle(order)
    for i in range(bin_num):
      shuffle(tmp[i])
    data = sum(tmp, [])
    for i in order:
      if i % num_workers == idx:
        yield list(zip(*data[i:(i+batch_size)]))
        
_random.seed(29520)
_random.shuffle(data)
        
n_train = int(len(data) * 0.85/params.batch_size) * params.batch_size
n_valid=n_train+int(len(data)*0.05)
n_test=n_valid+int(len(data)*0.1)
test_set=data[n_valid:n_test]
valid_set=data[n_train:n_valid]
batchs = {
      "train": batch_generator(data[:n_train], params.batch_size),
      "valid": batch_generator(data[n_train:n_valid], params.batch_size),
      "test": batch_generator(data[n_valid:n_test], params.batch_size)
  }




with tf.device(tf.train.replica_device_setter(worker_device="/gpu:0",
                                                 ps_device="/cpu:0",
                                                 ps_tasks=1)):



    inputs = tf.placeholder(tf.float32, [None, None,feat_len])
    input_len = tf.placeholder(tf.int32, [None])
    output = tf.sparse_placeholder(tf.int32)
    is_train = tf.placeholder(tf.bool)
    cost, deco, ler=acoustic_model(inputs, input_len, output, params, is_train)

    
    tf.summary.scalar("CTC", cost)
    tf.summary.scalar("LER", ler)
    summary = tf.summary.merge_all()
    global_step = tf.train.get_or_create_global_step()

    train_op = tf.train.AdamOptimizer(params.learning_rate).minimize(cost, global_step,name="train_op")


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                         allow_soft_placement=True,
                         log_device_placement=True)


model_path = _path.join("/home/suji93/check", "model")
saver = tf.train.Saver(tf.all_variables(),max_to_keep=500)

writers = {phase: tf.summary.FileWriter(path.join("./logs", phase)) for phase in ["train_sen2", "valid_sen2","test_sen2"]}


with tf.Session(config=config) as sess:
  try:
      saver.restore(sess, tf.train.latest_checkpoint("/home/suji93/check"))
      global_step=tf.train.get_global_step()
      train_op = tf.get_default_graph().get_tensor_by_name("train_op:0")
      step = sess.run(global_step)
  except ValueError:
      sess.run(tf.global_variables_initializer())
      step=0
  
  for i in range(10):
        cost_total=ler_total=0
        print("===== TRAINING =====")
        for _ in range(int(n_train/params.batch_size)):
            
            
            step,trainsum,_ = sess.run([global_step,summary,train_op], feed_dict=feed(list(next(batchs["train"])), True))
            writers["train_sen2"].add_summary(trainsum, step)
        vcost,vler = sess.run([cost,ler], feed_dict=feed(list(next(batchs["valid"])), False))
        print(vcost)
        print(vler)
        
        saver.save(sess, model_path, global_step=i)      
  
  print("===== TESTING =====")
  
  testsum,str_decoded,test_cost, test_ler=sess.run([summary,deco, cost, ler], feed_dict=feed(list(next(batchs["test"])), False))
  writers["test_sen2"].add_summary(testsum)
  print("LER: %f" %vler)
  for ss in zip(list(zip(*data[n_valid:n_test]))[1], decode(str_decoded)):
      print("    Original: %s" % ss[0])
      print("    Decoded:  %s" % ss[1])
      

  print("Run 'tensorboard --logdir=%s'"%writers["train_sen2"].get_logdir())



