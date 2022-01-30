# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tempfile
import time

from collections import Counter
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adadelta
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import server_lib
from tensorflow.python.training import training_util
from tensorflow.python.yunfan_ops import dynamic_embedding_ops as deo
from tensorflow.python.yunfan_ops import dynamic_feature_filter as dff

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


def get_size_info(sess, policy):
  if isinstance(policy, dff.FrequencyFilterPolicy):
    policy_status = policy.freq_var
  elif isinstance(policy, dff.ProbabilityFilterPolicy):
    policy_status = policy.tstp_var
  else:
    raise Exception('Unknown Policy')
  var_size = sess.run(policy.var.size())
  status_var_size = sess.run(policy_status.size())
  return var_size, status_var_size


def _get_devices():
  return ['/gpu:0' if test_util.is_gpu_available() else '/cpu:0']


def get_random_sparse_tensor(batch_size, sample_length):
  vals = random_ops.random_uniform((batch_size, sample_length),
                                   maxval=1000,
                                   dtype=dtypes.int64)

  ids = random_ops.random_uniform((batch_size, sample_length * 3),
                                  maxval=1000,
                                  dtype=dtypes.int64)
  ids = array_ops.reshape(ids, (-1, 3))
  vals = array_ops.reshape(vals, (-1,))
  st = sparse_tensor.SparseTensor(indices=ids,
                                  values=vals,
                                  dense_shape=[1, 1, 1])
  return st


def get_random_dense_tensor(batch_size, sample_length):
  dt = random_ops.random_uniform((batch_size, sample_length),
                                 maxval=1000,
                                 dtype=dtypes.int64)
  dt = array_ops.reshape(dt, (-1,))
  return dt


def create_frequency_filter_with_tensor(var_list, tensor_list, gstep):
  feature_filter = dff.FeatureFilter(var_list=var_list,
                                     policy=dff.FrequencyFilterPolicy)

  update_op = feature_filter.update(input_tensor_list=tensor_list)
  filter_list = feature_filter.filter(input_tensor_list=tensor_list,
                                      frequency_threshold=2)

  cur_list = control_flow_ops.cond(math_ops.equal
                                   (math_ops.mod(gstep, 10), 0),
                                   lambda: filter_list,
                                   lambda: tensor_list)
  if not isinstance(cur_list, list):
    cur_list = [cur_list]
  return feature_filter, cur_list, update_op


def create_probability_filter_with_tensor(var_list, tensor_list, gstep):
  feature_filter = dff.FeatureFilter(var_list=var_list,
                                     policy=dff.ProbabilityFilterPolicy)

  filter_list = feature_filter.filter(input_tensor_list=tensor_list,
                                      probability=0.5)
  cur_list = control_flow_ops.cond(math_ops.equal
                                   (math_ops.mod(gstep, 10), 0),
                                   lambda: filter_list,
                                   lambda: tensor_list)
  if not isinstance(cur_list, list):
    cur_list = [cur_list]
  update_op = feature_filter.update(input_tensor_list=cur_list)
  return feature_filter, cur_list, update_op


def build_graph(var_list, filter_func, tensor_list,
                gstep, batch_size=2, sample_length=3):
  opt0 = gradient_descent.GradientDescentOptimizer(0.1)
  opt_list = [opt0]

  emb_domain_list = list()
  do_step = state_ops.assign_add(gstep, 1)
  feature_filter, cur_list, update_op = \
      filter_func(var_list, tensor_list, gstep)

  for idx, var in enumerate(var_list):
    if isinstance(cur_list[idx], sparse_tensor.SparseTensor):
      _, _tw = deo.embedding_lookup_sparse(var,
                                           cur_list[idx],
                                           None,
                                           return_trainable=True)
    else:
      _, _tw = deo.embedding_lookup(var,
                                    cur_list[idx],
                                    None,
                                    return_trainable=True)

    _collapse = array_ops.reshape(_tw, (batch_size, -1))
    _logits = math_ops.reduce_sum(_collapse, axis=1)
    _logits = math_ops.cast(_logits, dtypes.float32)
    emb_domain_list.append(_logits)

  logits = math_ops.add_n(emb_domain_list)
  labels = array_ops.zeros((batch_size,), dtype=dtypes.float32)
  loss = math_ops.reduce_mean(
    nn_impl.sigmoid_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
    ))

  _train_ops = list()
  for _opt in opt_list:
    _train_ops.append(_opt.minimize(loss))
  with ops.control_dependencies(update_op):
    train_op = control_flow_ops.group(_train_ops)

  return cur_list, train_op, do_step, feature_filter


default_config = config_pb2.ConfigProto(
    allow_soft_placement=True,
    inter_op_parallelism_threads=2,
    intra_op_parallelism_threads=2,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


@test_util.deprecated_graph_mode_only
class FrequencyFilterPolicyCreateTest(test.TestCase):

  def test_create_status_exception_invalid_var(self):
    var = variable_scope.get_variable("dense_var",
                                      shape=[10, 10],
                                      use_resource=False)
    with self.assertRaises(TypeError):
      _ = dff.FrequencyFilterPolicy(var)

  def test_create_status_var_not_lookup(self):
    var = deo.get_variable('v1',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)
    status = dff.FrequencyFilterPolicy(var)
    self.assertIsNotNone(status.freq_var)


@test_util.deprecated_graph_mode_only
class ProbabilityFilterPolicyCreateTest(test.TestCase):

  def test_create_status_exception_invalid_var(self):
    var = variable_scope.get_variable("dense_var",
                                      shape=[10, 10],
                                      use_resource=False)
    with self.assertRaises(TypeError):
      _ = dff.ProbabilityFilterPolicy(var)

  def test_create_status_var_not_lookup(self):
    var = deo.get_variable('v1',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)
    status = dff.ProbabilityFilterPolicy(var)
    self.assertIsNotNone(status.tstp_var)


@test_util.deprecated_graph_mode_only
class FilterPolicyUpdateTest(test.TestCase):

  def update_with_no_embedding_check(self, status):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      vals = math_ops.range(0, 100, dtype=dtypes.int64)
      vals = array_ops.reshape(vals, (-1,))
      ids = math_ops.range(0, 300, dtype=dtypes.int64)
      ids = array_ops.reshape(ids, (-1, 3))

      st = sparse_tensor.SparseTensor(indices=ids,
                                      values=vals,
                                      dense_shape=[2, 3, 4])
      update_op0 = status.update(input_tensor=st)
      update_op1 = status.update(input_tensor=vals)
      sess.run(update_op0)
      sess.run(update_op1)

      var_size, status_var_size = \
          get_size_info(sess, status)

      self.assertAllEqual(var_size, 0)
      self.assertAllEqual(status_var_size, 100)

  def update_variable_size_check(self, status):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      batch_size = 2
      sample_length = 3
      vals_size = batch_size * sample_length

      st = array_ops.sparse_placeholder(dtype=np.int64, name='placeholder0')
      dt = array_ops.placeholder(dtype=np.int64, name='placeholder1')
      update_op0 = status.update(input_tensor=st)
      update_op1 = status.update(input_tensor=dt)

      status_set = {}
      n, MAX_ITER = 0, 100
      while n < MAX_ITER:
        values0 = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        values1 = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        indices = np.random.randint(0, 100,
                                    size=(vals_size, 3)).astype(np.int64)
        shape = np.array([1, 1, 1]).astype(np.int64)
        sess.run(update_op0, feed_dict={st: (indices, values0, shape)})
        sess.run(update_op1, feed_dict={dt: values1})

        x = set(values0)
        y = set(values1)
        if not status_set:
          status_set = x | y
        else:
          status_set |= (x | y)

        var_size, status_var_size = \
            get_size_info(sess, status)

        self.assertAllEqual(status_var_size, len(status_set))
        n += 1

  def test_update_with_no_embedding(self):
    var = deo.get_variable('sp_var',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)
    freq_status = dff.FrequencyFilterPolicy(var)
    prob_status = dff.ProbabilityFilterPolicy(var)
    self.update_with_no_embedding_check(freq_status)
    self.update_with_no_embedding_check(prob_status)

  def test_update_variable_size(self):
    var = deo.get_variable('sp_var',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)
    freq_status = dff.FrequencyFilterPolicy(var)
    prob_status = dff.ProbabilityFilterPolicy(var)
    self.update_variable_size_check(freq_status)
    self.update_variable_size_check(prob_status)


@test_util.deprecated_graph_mode_only
class FrequencyFilterPolicyFilterTest(test.TestCase):

  def filter_variable_empty_and_full_check(self, var, tensor, vals_size):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:

      status = dff.FrequencyFilterPolicy(var)
      update_op = status.update(input_tensor=tensor)
      filter_tensor0 = status.filter(input_tensor=tensor, frequency_threshold=0)
      filter_tensor1 = status.filter(input_tensor=tensor, frequency_threshold=1)
      filter_tensor2 = status.filter(input_tensor=tensor, frequency_threshold=10000)

      n, MAX_ITER = 0, 50
      while n < MAX_ITER:
        sess.run(update_op)
        all_tensor = sess.run(filter_tensor0)
        some_tensor = sess.run(filter_tensor1)
        none_tensor = sess.run(filter_tensor2)

        var_size, freq_var_size = \
            get_size_info(sess, status)

        self.assertAllGreater(freq_var_size, 0)
        if isinstance(tensor, sparse_tensor.SparseTensor):
          self.assertAllEqual([len(none_tensor.indices),
                               len(none_tensor.values)], [0, 0])
          self.assertGreaterEqual(vals_size, len(some_tensor.indices))
          self.assertGreaterEqual(vals_size, len(some_tensor.values))
          self.assertAllEqual([len(all_tensor.indices),
                               len(all_tensor.values)],
                              [vals_size, vals_size])
        else:
          self.assertEqual(len(none_tensor), 0)
          self.assertGreaterEqual(vals_size, len(some_tensor))
          self.assertEqual(len(all_tensor), vals_size)
        n += 1

  def filter_tensor_frequency_check(self, var):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      status = dff.FrequencyFilterPolicy(var)

      th = 3
      batch_size = 2
      sample_length = 3
      vals_size = batch_size * sample_length

      st = array_ops.sparse_placeholder(dtype=np.int64, name='placeholder1')
      update_op = status.update(input_tensor=st)
      filter_st = status.filter(input_tensor=st, frequency_threshold=th)

      dt = array_ops.placeholder(dtype=np.int64, name='placeholder2')
      filter_dt = status.filter(input_tensor=dt, frequency_threshold=th)

      value_list = []
      n, MAX_ITER = 0, 50
      while n < MAX_ITER:
        values0 = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        values1 = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        indices = np.random.randint(0, 100,
                                    size=(vals_size, 3)).astype(np.int64)
        shape = np.array([1, 1, 1]).astype(np.int64)
        sess.run(update_op, feed_dict={st: (indices, values0, shape)})
        some_st = sess.run(filter_st, feed_dict={st: (indices, values0, shape)})
        some_dt = sess.run(filter_dt, feed_dict={dt: values1})

        value_list += list(set(values0))
        counter = Counter(value_list)
        value_dict = dict(counter)

        for val in some_st.values:
          self.assertGreaterEqual(value_dict[val], th)
        for val in some_dt:
          self.assertGreaterEqual(value_dict[val], th)
        n += 1

  def test_filter_tensor_result(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      var = deo.get_variable('sp_var',
                             key_dtype=dtypes.int64,
                             value_dtype=dtypes.float32,
                             initializer=-1.,
                             dim=1)
      status = dff.FrequencyFilterPolicy(var)
      indices0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
                           [13, 14, 15], [16, 17, 18]],
                          dtype=np.int64)
      indices1 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18],
                           [19, 20, 21], [22, 23, 24], [25, 26, 27]],
                          dtype=np.int64)
      indices2 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18],
                           [19, 20, 21], [22, 23, 24], [25, 26, 27]],
                          dtype=np.int64)

      values0 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
      values1 = np.array([3, 4, 5, 6, 7, 8], dtype=np.int64)
      values2 = np.array([3, 4, 5, 6, 7, 8], dtype=np.int64)

      tensor0 = sparse_tensor.SparseTensor(indices=indices0,
                                           values=values0,
                                           dense_shape=[2, 3, 4])
      tensor1 = sparse_tensor.SparseTensor(indices=indices1,
                                           values=values1,
                                           dense_shape=[2, 3, 4])
      tensor2 = sparse_tensor.SparseTensor(indices=indices2,
                                           values=values2,
                                           dense_shape=[2, 3, 4])
      update_op0 = status.update(input_tensor=tensor0)
      update_op1 = status.update(input_tensor=tensor1)
      filter_st0 = status.filter(input_tensor=tensor2, frequency_threshold=2)
      sess.run(update_op0)
      sess.run(update_op1)
      st = sess.run(filter_st0)

      var_size, freq_var_size = \
          get_size_info(sess, status)
      self.assertAllEqual([var_size, freq_var_size], [0, 9])

      self.assertAllEqual([len(st.indices), len(st.values)], [3, 3])
      self.assertAllEqual(st.indices.tolist(), 
                          [[10, 11, 12], [13, 14, 15], [16, 17, 18]])
      self.assertAllEqual(st.values.tolist(), [3, 4, 5])

      tensor3 = array_ops.constant([6, 7, 8, 9, 10, 11], dtype=np.int64)
      tensor4 = array_ops.constant([0, 13, 14, 15, 16, 17], dtype=np.int64)
      tensor5 = array_ops.constant([0, 3, 6, 12, 15, 16], dtype=np.int64)

      update_op2 = status.update(input_tensor=tensor3)
      update_op3 = status.update(input_tensor=tensor4)
      filter_st1 = status.filter(input_tensor=tensor5, frequency_threshold=2)
      sess.run(update_op2)
      sess.run(update_op3)
      dt = sess.run(filter_st1)

      var_size, freq_var_size = \
          get_size_info(sess, status)
      self.assertAllEqual([var_size, freq_var_size], [0, 17])
      self.assertAllEqual(dt, [0, 3, 6])

  def test_filter_variable_empty_and_full(self):
    var = deo.get_variable('sp_var',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)

    batch_size = 2
    sample_length = 3
    vals_size = batch_size * sample_length
    st = get_random_sparse_tensor(batch_size, sample_length)
    self.filter_variable_empty_and_full_check(var, st, vals_size)

    dt = get_random_dense_tensor(batch_size, sample_length)
    self.filter_variable_empty_and_full_check(var, dt, vals_size)

  def test_filter_tensor_frequency(self):
    var = deo.get_variable('sp_var',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)
    self.filter_tensor_frequency_check(var)

  def test_filter_tensor_frequency_partitioned(self):
    def _partition_fn(keys, shard_num):
      return math_ops.cast(keys % shard_num, dtype=dtypes.int32)

    var = deo.get_variable("t330",
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           partitioner=_partition_fn,
                           devices=_get_devices() * 3,
                           initializer=2.0)
    self.filter_tensor_frequency_check(var)


@test_util.deprecated_graph_mode_only
class ProbabilityFilterPolicyFilterTest(test.TestCase):

  def filter_variable_empty_and_full_check(self, var, vals_size):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:

      status = dff.ProbabilityFilterPolicy(var)
      dt = array_ops.placeholder(dtype=np.int64, name='placeholder1')
      filter_tensor0 = status.filter(input_tensor=dt, probability=1.0)
      filter_tensor1 = status.filter(input_tensor=dt, probability=0.5)
      filter_tensor2 = status.filter(input_tensor=dt, probability=0.0)

      n, MAX_ITER = 0, 50
      while n < MAX_ITER:
        values = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        all_tensor = sess.run(filter_tensor0 , feed_dict={dt: values})
        some_tensor = sess.run(filter_tensor1, feed_dict={dt: values})
        none_tensor = sess.run(filter_tensor2, feed_dict={dt: values})
        self.assertAllEqual(values, all_tensor)
        self.assertEqual(len(none_tensor), 0)
        self.assertGreaterEqual(vals_size, len(some_tensor))
        n += 1

      st = array_ops.sparse_placeholder(dtype=np.int64, name='placeholder2')
      filter_tensor0 = status.filter(input_tensor=st, probability=1.0)
      filter_tensor1 = status.filter(input_tensor=st, probability=0.5)
      filter_tensor2 = status.filter(input_tensor=st, probability=0.0)

      n, MAX_ITER = 0, 50
      while n < MAX_ITER:
        values = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        indices = np.random.randint(0, 100,
                                    size=(vals_size, 3)).astype(np.int64)
        shape = np.array([1, 1, 1]).astype(np.int64)
        all_tensor = sess.run(filter_tensor0 , feed_dict={st: (indices, values, shape)})
        some_tensor = sess.run(filter_tensor1, feed_dict={st: (indices, values, shape)})
        none_tensor = sess.run(filter_tensor2, feed_dict={st: (indices, values, shape)})
        self.assertAllEqual(values, all_tensor.values)
        self.assertEqual(len(none_tensor.values), 0)
        self.assertGreaterEqual(vals_size, len(some_tensor.values))
        n += 1

  def filter_tensor_frequency_check(self, var):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      status = dff.ProbabilityFilterPolicy(var)

      th = 3
      batch_size = 2
      sample_length = 3
      vals_size = batch_size * sample_length

      st = array_ops.sparse_placeholder(dtype=np.int64, name='placeholder1')
      update_op = status.update(input_tensor=st)
      filter_st0 = status.filter(input_tensor=st, probability=0.0)
      filter_st1 = status.filter(input_tensor=st, probability=0.4)
      filter_st2 = status.filter(input_tensor=st, probability=1.0)

      dt = array_ops.placeholder(dtype=np.int64, name='placeholder2')
      filter_dt0 = status.filter(input_tensor=dt, probability=0.0)
      filter_dt1 = status.filter(input_tensor=dt, probability=0.8)
      filter_dt2 = status.filter(input_tensor=dt, probability=1.0)

      value_list = []
      n, MAX_ITER = 0, 50
      while n < MAX_ITER:
        values0 = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        values1 = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        indices = np.random.randint(0, 100,
                                    size=(vals_size, 3)).astype(np.int64)
        shape = np.array([1, 1, 1]).astype(np.int64)
        none_dt = sess.run(filter_dt0, feed_dict={dt: values1})
        some_dt = sess.run(filter_dt1, feed_dict={dt: values1})
        all_dt = sess.run(filter_dt2, feed_dict={dt: values1})
        none_st = sess.run(filter_st0, feed_dict={st: (indices, values0, shape)})
        some_st = sess.run(filter_st1, feed_dict={st: (indices, values0, shape)})
        all_st = sess.run(filter_st2, feed_dict={st: (indices, values0, shape)})
        set1 = set(some_st.values.tolist())
        set2 = set(none_st.values.tolist())
        
        self.assertEqual(set1>=set2, True)
        self.assertEqual(set(some_dt)>=set(none_dt), True)
        self.assertAllEqual(all_st.values.tolist(), values0)
        self.assertAllEqual(all_dt, values1)
        sess.run(update_op, feed_dict={st: (indices, values0, shape)})

        n += 1

  def test_filter_tensor_result(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      var = deo.get_variable('sp_var',
                             key_dtype=dtypes.int64,
                             value_dtype=dtypes.float32,
                             initializer=-1.,
                             dim=1)
      status = dff.ProbabilityFilterPolicy(var)
      indices0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
                           [13, 14, 15], [16, 17, 18]],
                          dtype=np.int64)
      indices1 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18],
                           [19, 20, 21], [22, 23, 24], [25, 26, 27]],
                          dtype=np.int64)

      values0 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
      values1 = np.array([5, 6, 7, 8, 9, 10], dtype=np.int64)

      tensor0 = sparse_tensor.SparseTensor(indices=indices0,
                                           values=values0,
                                           dense_shape=[2, 3, 4])
      tensor1 = sparse_tensor.SparseTensor(indices=indices1,
                                           values=values1,
                                           dense_shape=[2, 3, 4])
      update_op0 = status.update(input_tensor=tensor0)
      sess.run(update_op0)
      time.sleep(1)
      filter_st0 = status.filter(input_tensor=tensor1, probability=1.0)
      st = sess.run(filter_st0)
      self.assertAllEqual(st.values.tolist(), [5, 6, 7, 8, 9, 10])
      self.assertAllEqual(st.indices.tolist(), 
                          [[10, 11, 12], [13, 14, 15], [16, 17, 18],
                           [19, 20, 21], [22, 23, 24], [25, 26, 27]])
      filter_st0 = status.filter(input_tensor=tensor1, probability=0.0)
      st = sess.run(filter_st0)
      self.assertAllEqual(st.values.tolist(), [5])
      self.assertAllEqual(st.indices.tolist(), [[10, 11, 12]])

      filter_st0 = status.filter(input_tensor=tensor1, probability=0.5)
      st = sess.run(filter_st0)
      self.assertGreaterEqual(len(st.values), 1)
      self.assertGreaterEqual(len(st.indices), 1)


      update_op1 = status.update(input_tensor=tensor1)
      sess.run(update_op1)
      time.sleep(1)

      tensor2 = array_ops.constant([1, 3, 5, 7, 9], dtype=np.int64)
      filter_dt0 = status.filter(input_tensor=tensor2, probability=0.5)
      dt = sess.run(filter_dt0)
      self.assertAllEqual(dt, [1, 3, 5, 7, 9])

      tensor2 = array_ops.constant([9, 11, 12, 13, 14], dtype=np.int64)
      filter_dt0 = status.filter(input_tensor=tensor2, probability=0.8)
      dt = sess.run(filter_dt0)
      self.assertGreaterEqual(len(dt), 1)

  def test_filter_variable_empty_and_full(self):
    var = deo.get_variable('sp_var',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)

    batch_size = 2
    sample_length = 3
    vals_size = batch_size * sample_length
    self.filter_variable_empty_and_full_check(var, vals_size)

  def test_filter_tensor_frequency(self):
    var = deo.get_variable('sp_var',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)
    self.filter_tensor_frequency_check(var)

  def test_filter_tensor_frequency_partitioned(self):
    def _partition_fn(keys, shard_num):
      return math_ops.cast(keys % shard_num, dtype=dtypes.int32)

    var = deo.get_variable("t330",
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           partitioner=_partition_fn,
                           devices=_get_devices() * 3,
                           initializer=2.0)
    self.filter_tensor_frequency_check(var)


@test_util.deprecated_graph_mode_only
class FilterPolicyRestrictTest(test.TestCase):

  def get_all_tensors_filter_op(self, status, tensor):
    filter_op = None
    if isinstance(status, dff.FrequencyFilterPolicy):
      filter_op = status.filter(input_tensor=tensor, frequency_threshold=1) 
    elif isinstance(status, dff.ProbabilityFilterPolicy):
      filter_op = status.filter(input_tensor=tensor, probability=0.0) 
    else:
      raise Exception('Unknown Policy')
    return filter_op

  def restrict_table_timestamp_check(self, status):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      th = 3
      batch_size = 2
      sample_length = 3
      vals_size = batch_size * sample_length

      st = array_ops.sparse_placeholder(dtype=np.int64, name='placeholder1')
      update_op0 = status.update(input_tensor=st)

      dt = array_ops.placeholder(dtype=np.int64, name='placeholder2')
      update_op1 = status.update(input_tensor=dt)

      value_list = []
      time_dict = {}
      n, MAX_ITER = 0, 100
      while n < MAX_ITER:
        values0 = np.random.randint(0, 500, size=vals_size).astype(np.int64)
        values1 = np.random.randint(0, 500, size=vals_size).astype(np.int64)
        indices = np.random.randint(0, 500,
                                    size=(vals_size, 3)).astype(np.int64)
        shape = np.array([1, 1, 1]).astype(np.int64)
        sess.run(update_op0, feed_dict={st: (indices, values0, shape)})
        sess.run(update_op1, feed_dict={dt: values1})
        
        dict0 = dict.fromkeys(list(set(values0)), n * 2)
        dict1 = dict.fromkeys(list(set(values1)), n * 2 + 1)
        time_dict.update(dict0)
        time_dict.update(dict1)
        n += 1
        if n >= 98:
          time.sleep(1)
      sort_dict = sorted(time_dict.items(), \
          key=lambda item:item[1],reverse=True)
      top_list = [v[0] for v in sort_dict[:12]]
      tensor0 = array_ops.constant(top_list, dtype=np.int64)

      restrict_op0 = status.restrict(threshold=12)
      filter_op0 = self.get_all_tensors_filter_op(status, tensor0)

      sess.run(restrict_op0)
      filter_tensor0 = sess.run(filter_op0)
      self.assertAllEqual(filter_tensor0, tensor0)

      top_list = [v[0] for v in sort_dict[12:24]]
      tensor1 = array_ops.constant(top_list, dtype=np.int64)
      filter_op1 = self.get_all_tensors_filter_op(status, tensor1)
      filter_tensor1 = sess.run(filter_op1)
      self.assertEqual(len(filter_tensor0), 12)
      self.assertEqual(len(filter_tensor1), 0)

  def restrict_threshold_check(self, status):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      th = 3
      max_size = 60
      batch_size = 2
      sample_length = 3
      vals_size = batch_size * sample_length

      st = array_ops.sparse_placeholder(dtype=np.int64, name='placeholder1')
      update_op0 = status.update(input_tensor=st)

      dt = array_ops.placeholder(dtype=np.int64, name='placeholder2')
      update_op1 = status.update(input_tensor=dt)
      restrict_op0 = status.restrict(threshold=max_size)

      value_list = []
      time_dict = {}
      n, MAX_ITER = 0, 100
      while n < MAX_ITER:
        values0 = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        values1 = np.random.randint(0, 100, size=vals_size).astype(np.int64)
        indices = np.random.randint(0, 100,
                                    size=(vals_size, 3)).astype(np.int64)
        shape = np.array([1, 1, 1]).astype(np.int64)
        sess.run(update_op0, feed_dict={st: (indices, values0, shape)})
        sess.run(update_op1, feed_dict={dt: values1})
        _, freq_var_size = get_size_info(sess, status)
        n += 1
        try:
          self.assertGreater(freq_var_size, max_size)
          break
        except AssertionError:
          continue
      _, freq_var_size = get_size_info(sess, status)
      self.assertGreater(freq_var_size, max_size)
      sess.run(restrict_op0)
      _, freq_var_size = get_size_info(sess, status)
      self.assertLessEqual(freq_var_size, max_size)

  def test_restrict_tensor_result(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      var = deo.get_variable('sp_var',
                             key_dtype=dtypes.int64,
                             value_dtype=dtypes.float32,
                             initializer=-1.,
                             dim=1)
      status = dff.FrequencyFilterPolicy(var)

      vals = math_ops.range(0, 5, dtype=dtypes.int64)
      vals = array_ops.reshape(vals, (-1,))
      ids = math_ops.range(0, 15, dtype=dtypes.int64)
      ids = array_ops.reshape(ids, (-1, 3))

      st = sparse_tensor.SparseTensor(indices=ids,
                                      values=vals,
                                      dense_shape=[2, 3, 4])
      update_op0 = status.update(input_tensor=st)
      sess.run(update_op0)
      time.sleep(1)

      vals1 = math_ops.range(5, 10, dtype=dtypes.int64)
      update_op1 = status.update(input_tensor=vals1)
      sess.run(update_op1)
      time.sleep(1)

      restrict_op0 = status.restrict(threshold=100)
      filter_op0 = status.filter(input_tensor=vals, frequency_threshold=1)
      sess.run(restrict_op0)
      filter_tensor0 = sess.run(filter_op0)
      self.assertAllEqual(filter_tensor0, vals)

      restrict_op1 = status.restrict(threshold=5)
      sess.run(restrict_op1)
      vals2 = math_ops.range(0, 10, dtype=dtypes.int64)
      filter_op1 = status.filter(input_tensor=vals2, frequency_threshold=1)
      filter_tensor1 = sess.run(filter_op1)
      self.assertAllEqual(filter_tensor1, [5, 6, 7, 8, 9])

      vals3 = math_ops.range(7, 15, dtype=dtypes.int64)
      update_op3 = status.update(input_tensor=vals3)
      restrict_op3 = status.restrict(threshold=8)
      filter_op3 = status.filter(input_tensor=vals1, frequency_threshold=1)
      sess.run(update_op3)
      time.sleep(1)
      sess.run(restrict_op3)
      filter_tensor3 = sess.run(filter_op3)
      self.assertAllEqual(filter_tensor3, [7, 8, 9])

  def test_restrict_table_timestamp(self):
    var = deo.get_variable('sp_var',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)
    status = dff.FrequencyFilterPolicy(var)
    self.restrict_table_timestamp_check(status)

    status = dff.ProbabilityFilterPolicy(var)
    self.restrict_table_timestamp_check(status)

  def test_restrict_threshold(self):
    var = deo.get_variable('sp_var',
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           initializer=-1.,
                           dim=1)
    status = dff.FrequencyFilterPolicy(var)
    self.restrict_threshold_check(status)

    status = dff.ProbabilityFilterPolicy(var)
    self.restrict_threshold_check(status)

  def test_restrict_threshold_partitioned(self):
    def _partition_fn(keys, shard_num):
      return math_ops.cast(keys % shard_num, dtype=dtypes.int32)

    var = deo.get_variable("t330",
                           key_dtype=dtypes.int64,
                           value_dtype=dtypes.float32,
                           partitioner=_partition_fn,
                           devices=_get_devices() * 4,
                           initializer=2.0)
    status = dff.FrequencyFilterPolicy(var)
    self.restrict_threshold_check(status)

    status = dff.ProbabilityFilterPolicy(var)
    self.restrict_threshold_check(status)


@test_util.deprecated_graph_mode_only
class FeatureFilterTest(test.TestCase):

  def train_result_check(self, n, cur_list, batch_size, sample_length):
    for cur_tensor in cur_list:
      if n % 10 == 0:
        if isinstance(cur_tensor, sparse_tensor.SparseTensor):
          self.assertGreaterEqual(batch_size * sample_length,
                                  len(cur_tensor.eval().indices))
          self.assertGreaterEqual(batch_size * sample_length,
                                  len(cur_tensor.eval().values))
        else:
          self.assertGreaterEqual(batch_size * sample_length,
                                  len(cur_tensor.eval()))
      else:
        if isinstance(cur_tensor, sparse_tensor.SparseTensor):
          self.assertEqual(batch_size * sample_length,
                           len(cur_tensor.eval().indices))
          self.assertEqual(batch_size * sample_length,
                           len(cur_tensor.eval().values))
        else:
          self.assertEqual(batch_size * sample_length,
                           len(cur_tensor.eval()))

  def restrict_table_check(self, sess, feature_filter, max_size):
    restrict_op = feature_filter.restrict(threshold=max_size)
    sess.run(restrict_op)
    for idx, pol in enumerate(feature_filter.policy_list):
      _, freq_var_size = \
          get_size_info(sess, pol)
      self.assertLessEqual(freq_var_size, max_size)

  def run_feature_filter_single_var_check(self, var_list,
                                               opt_list, filter_func,
                                               gstep, tensor_list,
                                               batch_size, sample_length):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      emb_domain_list = list()
      do_step = state_ops.assign_add(gstep, 1)
      feature_filter, cur_list, update_op = \
          filter_func(var_list, tensor_list, gstep)

      if isinstance(cur_list[0], sparse_tensor.SparseTensor):
        _, _tw = deo.embedding_lookup_sparse(var_list[0],
                                             cur_list[0],
                                             None,
                                             return_trainable=True)
      else:
        _, _tw = deo.embedding_lookup(var_list[0],
                                      cur_list[0],
                                      None,
                                      return_trainable=True)
      _collapse = array_ops.reshape(_tw, (batch_size, -1))
      _logits = math_ops.reduce_sum(_collapse, axis=1)
      _logits = math_ops.cast(_logits, dtypes.float32)
      emb_domain_list.append(_logits)
      logits = math_ops.add_n(emb_domain_list)

      labels = array_ops.zeros((batch_size,), dtype=dtypes.float32)
      loss = math_ops.reduce_mean(
        nn_impl.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
        ))

      _train_ops = list()
      for _opt in opt_list:
        _train_ops.append(_opt.minimize(loss))
      with ops.control_dependencies(update_op):
        train_op = control_flow_ops.group(_train_ops)

      self.evaluate(variables.global_variables_initializer())
      n, MAX_ITER = 0, 100
      while n < MAX_ITER:
        self.train_result_check(n, cur_list, batch_size, sample_length)
        sess.run(train_op)
        sess.run(do_step)
        n += 1

      max_size = 200
      self.restrict_table_check(sess, feature_filter, max_size)

  def run_feature_filter_multi_vars_check(self, var_list, filter_func,
                                               tensor_list, batch_size, 
                                               sample_length, gstep):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      cur_list, train_op, do_step, feature_filter \
          = build_graph(var_list, filter_func, tensor_list, gstep)
      self.evaluate(variables.global_variables_initializer())
      n, MAX_ITER = 0, 100
      while n < MAX_ITER:
        self.assertEqual(len(cur_list), len(var_list))
        self.train_result_check(n, cur_list, batch_size, sample_length)
        sess.run(train_op)
        sess.run(do_step)
        n += 1

      max_size = 200
      self.restrict_table_check(sess, feature_filter, max_size)

  def run_filter_restore_from_checkpoint_check(self, var_list, filter_func,
                                               tensor_list, batch_size, 
                                               sample_length, gstep,
                                               save_path):
    save_freq_size0 = 0
    save_freq_size1 = 0
    cur_list, train_op, do_step, feature_filter \
        = build_graph(var_list, filter_func, tensor_list, gstep)
    saver = saver_lib.Saver()
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      self.evaluate(variables.global_variables_initializer())
      n, MAX_ITER = 0, 100
      while n < MAX_ITER:
        self.assertEqual(len(cur_list), len(var_list))
        self.train_result_check(n, cur_list, batch_size, sample_length)

        sess.run(train_op)
        sess.run(do_step)
        _, freq_var_size0 = \
            get_size_info(sess, feature_filter.policy_list[0])
        _, freq_var_size1 = \
            get_size_info(sess, feature_filter.policy_list[1])
        save_freq_size0 = freq_var_size0
        save_freq_size1 = freq_var_size1
        n += 1
      max_size = 100
      self.restrict_table_check(sess, feature_filter, max_size)
      rt_save_path = saver.save(sess, save_path)
      self.assertAllEqual(rt_save_path, save_path)

    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      self.evaluate(variables.global_variables_initializer())
      _, freq_var_size0 = \
          get_size_info(sess, feature_filter.policy_list[0])
      _, freq_var_size1 = \
          get_size_info(sess, feature_filter.policy_list[1])
      self.assertEqual(0, freq_var_size0)
      self.assertEqual(0, freq_var_size1)
      saver.restore(sess, save_path)
      self.train_result_check(n, cur_list, batch_size, sample_length)

      sess.run(train_op)
      sess.run(do_step)
      _, freq_var_size0 = \
          get_size_info(sess, feature_filter.policy_list[0])
      _, freq_var_size1 = \
          get_size_info(sess, feature_filter.policy_list[1])
      self.assertGreaterEqual(freq_var_size0, 100)
      self.assertGreaterEqual(freq_var_size1, 100)

  def run_feature_filter_distributed_check(self, filter_func, gstep, name):
    batch_size = 2
    sample_length = 3
    vals_size = batch_size * sample_length
    emb_domain_list = list()
    do_step = state_ops.assign_add(gstep, 1)

    server0 = server_lib.Server.create_local_server()
    server1 = server_lib.Server.create_local_server()
    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server0.target[len('grpc://'):]
    job.tasks[1] = server1.target[len('grpc://'):]

    config = config_pb2.ConfigProto(
        cluster_def=cluster_def,
        experimental=config_pb2.ConfigProto.Experimental(
            share_session_state_in_clusterspec_propagation=True, ),
    )
    config.allow_soft_placement = True

    with ops.device('/job:worker/task:0'):
      var0 = deo.get_variable('sp_var'+name,
                              devices=[
                                  '/job:worker/task:1',
                              ],
                              initializer=0.0,
                              dim=2)
      var1 = deo.get_variable('sp_var1'+name,
                              devices=[
                                  '/job:worker/task:1',
                              ],
                              initializer=0.0,
                              dim=2)
      var2 = deo.get_variable('sp_var2'+name,
                              devices=[
                                  '/job:worker/task:1',
                              ],
                              initializer=0.0,
                              dim=2)
      var_list = [var0, var1, var2]
      opt0 = gradient_descent.GradientDescentOptimizer(0.1)
      opt_list = [opt0]

      st_list = []
      for var in var_list:
        st = get_random_sparse_tensor(batch_size, sample_length)
        st_list.append(st)

      feature_filter, cur_st_list, update_op = \
          filter_func(var_list, st_list, gstep)

      for idx, var in enumerate(var_list):
        _, _tw = deo.embedding_lookup_sparse(var,
                                             cur_st_list[idx],
                                             None,
                                             return_trainable=True)
        _collapse = array_ops.reshape(_tw, (batch_size, -1))
        _logits = math_ops.reduce_sum(_collapse, axis=1)
        _logits = math_ops.cast(_logits, dtypes.float32)
        emb_domain_list.append(_logits)

      logits = math_ops.add_n(emb_domain_list)
      labels = array_ops.zeros((batch_size,), dtype=dtypes.float32)
      loss = math_ops.reduce_mean(
          nn_impl.sigmoid_cross_entropy_with_logits(
              logits=logits,
              labels=labels,
          ))

      _train_ops = list()
      for _opt in opt_list:
        _train_ops.append(_opt.minimize(loss))
      with ops.control_dependencies(update_op):
        train_op = control_flow_ops.group(_train_ops)

      with session.Session(server0.target, config=config) as sess:
        self.evaluate(variables.global_variables_initializer())
        n, MAX_ITER = 0, 100
        while n < MAX_ITER:
          self.assertEqual(len(cur_st_list), len(var_list))
          for cur_st in cur_st_list:
            if n % 10 == 0:
              self.assertGreaterEqual(batch_size * sample_length,
                                      len(cur_st.eval().indices))
              self.assertGreaterEqual(batch_size * sample_length,
                                      len(cur_st.eval().values))
            else:
              self.assertEqual(batch_size * sample_length,
                               len(cur_st.eval().indices))
              self.assertEqual(batch_size * sample_length,
                               len(cur_st.eval().values))

          sess.run(train_op)
          sess.run(do_step)
          n += 1
        max_size = 200
        self.restrict_table_check(sess, feature_filter, max_size)

  def test_init_exception_invalid_policy(self):
    var_list = [
      deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    err = None
    with self.assertRaises(TypeError):
      _ = dff.FeatureFilter(var_list=var_list,
                            policy=None)

  def test_run_feature_filter_single_var(self):
    var_list = [
      deo.get_variable('sp_var', initializer=0.0, dim=2),
    ]
    opt_list = [
      adadelta.AdadeltaOptimizer(),
    ]
    batch_size = 2
    sample_length = 3
    gstep = training_util.create_global_step()
    
    func = create_frequency_filter_with_tensor
    st = get_random_sparse_tensor(batch_size, sample_length)
    self.run_feature_filter_single_var_check(var_list, opt_list,
                                                  func, gstep, [st],
                                                  batch_size, sample_length)

    func = create_probability_filter_with_tensor
    dt = get_random_dense_tensor(batch_size, sample_length)
    self.run_feature_filter_single_var_check(var_list, opt_list,
                                                  func, gstep, [dt],
                                                  batch_size, sample_length)

  def test_run_feature_filter_multi_vars(self):
    var0 = deo.get_variable('sp_var', initializer=0.0, dim=2)
    var1 = deo.get_variable('sp_var1', initializer=0.0, dim=2)
    var2 = deo.get_variable('sp_var2', initializer=0.0, dim=2)
    var_list = [var0, var1, var2]
    batch_size = 2
    sample_length = 3
    sparse_tensor_list = []
    dense_tensor_list = []
    gstep = training_util.create_global_step()

    for _ in var_list:
      st = get_random_sparse_tensor(batch_size, sample_length)
      sparse_tensor_list.append(st)
    for _ in var_list:
      dt = get_random_dense_tensor(batch_size, sample_length)
      dense_tensor_list.append(dt)

    func = create_frequency_filter_with_tensor
    self.run_feature_filter_multi_vars_check(var_list, func,
                                                  sparse_tensor_list,
                                                  batch_size, sample_length,
                                                  gstep)
    func = create_probability_filter_with_tensor
    self.run_feature_filter_multi_vars_check(var_list, func,
                                                  dense_tensor_list,
                                                  batch_size, sample_length,
                                                  gstep)

  def test_run_feature_filter_restore_from_checkpoint(self):
    var0 = deo.get_variable('sp_var', initializer=0.0, dim=2)
    var1 = deo.get_variable('sp_var1', initializer=0.0, dim=2)
    var_list = [var0, var1]
    batch_size = 2
    sample_length = 3
    sparse_tensor_list = []
    dense_tensor_list = []
    gstep = training_util.create_global_step()
    save_dir = os.path.join(self.get_temp_dir(), 'save_restore')

    for _ in var_list:
      st = get_random_sparse_tensor(batch_size, sample_length)
      sparse_tensor_list.append(st)
    for _ in var_list:
      dt = get_random_dense_tensor(batch_size, sample_length)
      dense_tensor_list.append(dt)
    func = create_frequency_filter_with_tensor
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), 'filter')
    self.run_filter_restore_from_checkpoint_check(var_list, func,
                                                  sparse_tensor_list,
                                                  batch_size, sample_length,
                                                  gstep, save_path)
    func = create_probability_filter_with_tensor
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), 'filter')
    self.run_filter_restore_from_checkpoint_check(var_list, func,
                                                  dense_tensor_list,
                                                  batch_size, sample_length,
                                                  gstep, save_path)

  def test_run_freq_feature_filter_with_default_value(self):
    var0 = deo.get_variable('sp_var', initializer=0.0, dim=2)
    var1 = deo.get_variable('sp_var1', initializer=0.0, dim=2)
    var_list = [var0, var1]
    default_value_list = [0, 10]
    opt0 = gradient_descent.GradientDescentOptimizer(0.1)
    opt_list = [opt0]
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()) as sess:
      feature_filter = dff.FeatureFilter(var_list=var_list,
                                         default_value_list=default_value_list,
                                         policy=dff.FrequencyFilterPolicy)

      indices0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
                           [13, 14, 15], [16, 17, 18]], dtype=np.int64)
      indices1 = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18],
                           [19, 20, 21], [22, 23, 24], [25, 26, 27]],
                          dtype=np.int64)
      values0 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
      values1 = np.array([3, 4, 5, 6, 7, 8], dtype=np.int64)

      tensor0 = sparse_tensor.SparseTensor(indices=indices0,
                                           values=values0,
                                           dense_shape=[2, 3, 4])
      tensor1 = sparse_tensor.SparseTensor(indices=indices1,
                                           values=values1,
                                           dense_shape=[2, 3, 4])
      st_list = [tensor0, tensor1]
      update_op = feature_filter.update(input_tensor_list=st_list)
      filter_st = feature_filter.filter(input_tensor_list=st_list,
                                        frequency_threshold=10)
      sess.run(update_op)
      tensor_list = sess.run(filter_st)

      self.assertAllEqual(
        [len(tensor_list[0].indices),
         len(tensor_list[0].values),
         len(tensor_list[1].indices)],
        [0, 0, 6])
      self.assertAllEqual(tensor_list[1].values,
                          [3, 4, 5, 6, 7, 8])

      tensor2 = array_ops.constant([0, 1, 2, 3, 4, 5], dtype=np.int64)
      tensor3 = array_ops.constant([3, 4, 5, 6, 7, 8], dtype=np.int64)
      dt_list = [tensor2, tensor3]
      filter_dt = feature_filter.filter(input_tensor_list=dt_list,
                                        frequency_threshold=10)
      tensor_list = sess.run(filter_dt)
      self.assertEqual(len(tensor_list[0]), 0)
      self.assertAllEqual(tensor_list[1],
                          [3, 4, 5, 6, 7, 8])

  def test_run_feature_filter_distributed(self):
    gstep = training_util.create_global_step()
    func = create_frequency_filter_with_tensor
    self.run_feature_filter_distributed_check(func, gstep, "freq")

    func = create_probability_filter_with_tensor
    self.run_feature_filter_distributed_check(func, gstep, "prob")


if __name__ == '__main__':
  os.environ['TF_HASHTABLE_INIT_SIZE'] = str(64 * 1024)
  test.main()

