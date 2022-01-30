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
"""Filter of variables"""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.yunfan_ops import dynamic_embedding_ops


@tf_export("dynamic_embedding.FilterPolicy")
class FilterPolicy(object):
  """
  FilterPolicy records the status of variable, while provides
  interfaces for continuously updating with the status with training
  progress, and filtering the eligible sparse tensor keys for training.
  The FilterPolicy is an abstract class, which can be inherited
  for customization.

  FilterPolicy holds `create_status`, `update`, `filter` and 'restrict' methods:
    create_status: creating the representation of the status.
    update: updating the status in iteration. The update operation usually
      runs before the training operation.
    filter: filtering sparse tensor keys for training according to
      the status. It's used to prohibit some feature keys from training.
    restrict: restricting the status table size to prevent the over growth of
      memory usage.
  """
  def __init__(self, var):
    """
    Construct the FilterPolicy from variable.

    Args:
      var: dynamic_ebmedding.Variable.
    """
    if not isinstance(var, dynamic_embedding_ops.Variable):
      raise TypeError("parameter var type should be" \
                      "dynamic_embedding_ops.Variable.")

    self.var = var
    self.threshold = 0
    self.create_status()

  def create_status(self, **kwargs):
    """
    Create status for recording the variable.
    """
    raise NotImplementedError

  def update(self, **kwargs):
    """
    Update the status. The returned update operation is
    usually used with training operation, to keep the status
    following change of variables.

    Returns:
      An operation to update the status.
    """
    raise NotImplementedError

  def filter(self, **kwargs):
    """
    filter the feature keys following the direction by records
    of status.

    Returns:
      A list of feature keys that filter for training.
    """
    raise NotImplementedError

  def restrict(self, **kwargs):
    """
    Restrict the status table size to prevent out-of-memory
    when status table size is growing.

    Returns:
      An operation to restrict the status.
    """
    raise NotImplementedError


@tf_export("dynamic_embedding.FrequencyFilterPolicy")
class FrequencyFilterPolicy(FilterPolicy):
  """
  A status inherts from FilterPolicy, providing
  updating and filtering for variable by frequency rule.

  When call filter method, the class will filter values on
  ids by following eliminated-below-threshold rule for every ids
  in record. And when call update method, the record of every
  ids will be increased by 1.
  """

  def __init__(self, var, **kwargs):
    default_count = kwargs.get('default_value', 0)
    self.default_count = constant_op.constant([default_count, 0], dtypes.int32)
    super(FrequencyFilterPolicy, self).__init__(var)

  def create_status(self, **kwargs):
    """
    Create relative frequency status variables.
    """
    scope = variable_scope.get_variable_scope()
    if scope.name:
      scope_name = scope.name + '/frequency_status_for_filter'
    else:
      scope_name = 'frequency_status_for_filter'

    with ops.name_scope(scope_name, "frequency_status_for_filter",
                        []) as unique_scope:
      full_name = unique_scope + '/' + self.var.name
      self.freq_var = dynamic_embedding_ops.get_variable(
          key_dtype=self.var.key_dtype,
          value_dtype=dtypes.int32,
          dim=2,
          name=full_name,
          devices=self.var.devices,
          partitioner=self.var.partition_fn,
          initializer=self.default_count,
          trainable=False,
          init_size=self.var.init_size,
          checkpoint=self.var.checkpoint,
          checkpoint_path=self.var.checkpoint_path
      )

  def update(self, input_tensor=None, **kwargs):
    """
    Update the frequency status. The corresponding frequency
    records will be increased by 1.

    Args:
      **kwargs: keyword arguments, including
      input_tensor: SparseTensor or dense tensor.
                    Feature keys need to count.

    Returns:
      An operation for updating the frequency status.
    """
    maxint32 = 2147483647
    update_ops = []

    if input_tensor is None:
      raise KeyError("update method expects parameter `input_tensor`.")
    elif isinstance(input_tensor, ops.Tensor):
      values = input_tensor
    elif isinstance(input_tensor, sparse_tensor.SparseTensor):
      values = input_tensor.values
    else:
      raise TypeError("input_tensor must be " \
                "either a SparseTensor or dense Tensor.")

    values, _ = array_ops.unique(values)
    status_values = array_ops.reshape(values, (-1,))
    partition_index = \
        self.var.partition_fn(status_values, self.var.shard_num)
    partitioned_values_list, partitioned_indices_list = \
        dynamic_embedding_ops._partition(status_values,
                                         partition_index,
                                         self.var.shard_num)

    for idx, dev in enumerate(self.freq_var.devices):
      with ops.device(dev):
        feature_status = \
            self.freq_var.tables[idx].lookup(
                partitioned_values_list[idx],
                dynamic_default_values=self.default_count,
            )
        feature_counts = array_ops.slice(feature_status, [0, 0], [-1, 1])
        feature_tstps = array_ops.slice(feature_status, [0, 1], [-1, 1])
        feature_tstps = array_ops.tile(
            array_ops.reshape(gen_logging_ops.timestamp(), [1]),
            array_ops.reshape(array_ops.size(feature_counts), (-1,)),
        )
        feature_tstps = math_ops.cast(feature_tstps, dtypes.int32)
        feature_tstps = array_ops.reshape(feature_tstps, (-1, 1))

        condition = math_ops.less(feature_counts, maxint32)
        feature_counts = array_ops.where(condition, feature_counts + 1,
                                         feature_counts)

        feature_status = array_ops.concat([feature_counts, feature_tstps], 1)

        mht_update = \
            self.freq_var.tables[idx].insert(
                partitioned_values_list[idx],
                feature_status,
            )
        update_ops.append(mht_update)

    return control_flow_ops.group(update_ops)

  def filter(self, input_tensor=None, **kwargs):
    """
    Filter feature keys below the threshold before training.
    Prevent unpopular feature keys from affecting training.

    Args:
      **kwargs: keyword arguments, including
      input_tensor: SparseTensor or DenseTensor.
                    Feature keys need to filter.
      frequency_threshold: int. Filtering feature keys whose frequency values
                      are less than the threshold.

    Returns:
      Tensor that are filtered for training.
    """

    if input_tensor is None:
      raise KeyError("filter method expects parameter `input_tensor`.")
    elif isinstance(input_tensor, ops.Tensor):
      input_type = "DenseTensor"
      values = input_tensor
    elif isinstance(input_tensor, sparse_tensor.SparseTensor):
      input_type = "SparseTensor"
      values = input_tensor.values
      indices = input_tensor.indices
    else:
      raise TypeError("input_tensor must be " \
                "either a SparseTensor or dense Tensor.")

    if 'frequency_threshold' in kwargs:
      frequency_threshold = kwargs['frequency_threshold']
    else:
      raise KeyError("filter method expects parameter `frequency_threshold`.")
    if not isinstance(frequency_threshold, int):
      raise TypeError("frequency_threshold must be an integer.")
    if frequency_threshold < 0:
      raise ValueError("frequency_threshold must be greater or equal to zero.")

    unique_values, value_idx = array_ops.unique(values)
    status_values = array_ops.reshape(unique_values, (-1,))
    partition_index = \
        self.var.partition_fn(status_values, self.var.shard_num)
    partitioned_values_list, partitioned_indices_list = \
        dynamic_embedding_ops._partition(status_values,
                                         partition_index,
                                         self.var.shard_num)

    mask = []
    for idx, dev in enumerate(self.freq_var.devices):
      with ops.device(dev):
        feature_status = \
            self.freq_var.tables[idx].lookup(
                partitioned_values_list[idx],
                dynamic_default_values=self.default_count,
            )

        feature_counts = array_ops.slice(feature_status, [0, 0], [-1, 1])
        sub_fv = array_ops.reshape(feature_counts, (-1,))
        partitioned_mask = math_ops.greater_equal(sub_fv, frequency_threshold)
        mask.append(partitioned_mask)

    total_mask = dynamic_embedding_ops._stitch(mask, partitioned_indices_list)
    total_mask = array_ops.gather(total_mask, value_idx)
    total_mask.set_shape([None])
    filter_values = array_ops.boolean_mask(values, total_mask)
    if input_type == "DenseTensor":
      filter_tensor = filter_values
    elif input_type == "SparseTensor":
      filter_indices = array_ops.boolean_mask(indices, total_mask)
      filter_tensor = sparse_tensor.SparseTensor(
          indices=filter_indices,
          values=filter_values,
          dense_shape=input_tensor.dense_shape)

    return filter_tensor

  def restrict(self, **kwargs):
    """
    Restrict the status table size, eliminate the oldest
    feature keys, if the size of variable grow too large for
    threshold.

    Args:
      **kwargs: Keyword arguments, including
      threshold: int. The threshold for feature number
        in variable. When restrict method is called, the table
        size will be reduced to 'threshold'.
      factor: int,float,tf.int32,tf.int64,tf.float32.
        If the table size is greater than threshold * factor,
        restricting wiil be triggered.

    Returns:
      An operation to restrict the size of variable.
    """
    try:
      self.threshold = kwargs['threshold']
    except:
      raise KeyError("restrict method expects parameter `threshold`.")
    if not isinstance(self.threshold, int):
      raise TypeError("threshold must be an integer.")
    if self.threshold < 0:
      raise ValueError("threshold must be greater or equal to zero.")

    factor = kwargs.get('factor', 1.0)
    if isinstance(factor, ops.Tensor):
      if factor.dtype not in (dtypes.int32, dtypes.int64, dtypes.float32):
        raise TypeError(
            'factor expects int, float, tf.int32, tf.int64, or tf.float32')
      factor = math_ops.cast(factor, dtype=dtypes.float32)
    if not isinstance(factor, (int, float)):
      raise TypeError(
          'factor expects int, float, tf.int32, tf.int64, or tf.float32')

    cond_size = math_ops.cast(self.threshold, dtype=dtypes.float32) * factor
    cond_size = math_ops.cast(cond_size, dtype=dtypes.int64)
    condition = math_ops.greater(self.freq_var.size(), cond_size)
    restrict_status_ops = list()
    no_ops = list()

    for idx, dev in enumerate(self.freq_var.devices):
      with ops.device(dev):
        sub_tk, sub_tv = self.freq_var.tables[idx].export()
        sharded_threshold = int(self.threshold / self.freq_var.shard_num)

        sub_tv = array_ops.slice(sub_tv, [0, 1], [-1, 1])
        sub_tv = array_ops.reshape(sub_tv, (-1,))
        first_dim = array_ops.shape(sub_tv)[0]

        k_on_top = math_ops.cast(first_dim - sharded_threshold,
                                 dtype=dtypes.int32)
        k_on_top = math_ops.maximum(k_on_top, 0)
        _, removed_keys_ids = nn_ops.top_k(-sub_tv, k_on_top, sorted=False)
        removed_keys = array_ops.gather(sub_tk, removed_keys_ids)
        restrict_status_ops.append(
            self.freq_var.tables[idx].remove(removed_keys))
        no_ops.append(control_flow_ops.no_op())
    restrict_op = control_flow_ops.cond(condition, 
                                        lambda:restrict_status_ops,
                                        lambda:no_ops)

    return restrict_op


@tf_export("dynamic_embedding.ProbabilityFilterPolicy")
class ProbabilityFilterPolicy(FilterPolicy):
  """
  A status inherts from FilterPolicy, providing
  updating and filtering for variable by probability rule.

  When call filter method, the class will filter values on
  ids by following probability rule for new ids (no recorded
  in the table). And when call update method, new ids will
  be stored in the table.
  """

  def __init__(self, var, **kwargs):
    self.default_tstp = constant_op.constant(0, dtypes.int32)
    super(ProbabilityFilterPolicy, self).__init__(var)

  def create_status(self, **kwargs):
    """
    Create relative probability status variables.
    """
    scope = variable_scope.get_variable_scope()
    if scope.name:
      scope_name = scope.name + '/probability_status_for_filter'
    else:
      scope_name = 'probability_status_for_filter'

    with ops.name_scope(scope_name, "probability_status_for_filter",
                        []) as unique_scope:
      full_name = unique_scope + '/' + self.var.name
      self.tstp_var = dynamic_embedding_ops.get_variable(
          key_dtype=self.var.key_dtype,
          value_dtype=dtypes.int32,
          dim=1,
          name=full_name,
          devices=self.var.devices,
          partitioner=self.var.partition_fn,
          initializer=self.default_tstp,
          trainable=False,
          init_size=self.var.init_size,
          checkpoint=self.var.checkpoint,
          checkpoint_path=self.var.checkpoint_path
      )

  def update(self, input_tensor=None, **kwargs):
    """
    Update the probability status table. The filter ids will be
    stored in the table and record timestamp.

    Args:
      **kwargs: keyword arguments, including
      input_tensor: SparseTensor or dense tensor.
                    Feature keys need to count.

    Returns:
      An operation for updating the frequency status.
    """
    update_ops = []

    if input_tensor is None:
      raise KeyError("update method expects parameter `input_tensor`.")
    elif isinstance(input_tensor, ops.Tensor):
      values = input_tensor
    elif isinstance(input_tensor, sparse_tensor.SparseTensor):
      values = input_tensor.values
    else:
      raise TypeError("input_tensor must be " \
                "either a SparseTensor or dense Tensor.")

    values, _ = array_ops.unique(values)
    status_values = array_ops.reshape(values, (-1,))
    partition_index = \
        self.var.partition_fn(status_values, self.var.shard_num)
    partitioned_values_list, partitioned_indices_list = \
        dynamic_embedding_ops._partition(status_values,
                                         partition_index,
                                         self.var.shard_num)

    for idx, dev in enumerate(self.tstp_var.devices):
      with ops.device(dev):
        value_size = array_ops.size(partitioned_values_list[idx])
        feature_tstps = array_ops.tile(
            array_ops.reshape(gen_logging_ops.timestamp(), [1]),
            array_ops.reshape(value_size, (-1,)),
        )
        feature_tstps  = math_ops.cast(feature_tstps, dtypes.int32)
        feature_status = array_ops.reshape(feature_tstps, (-1, 1))

        mht_update = \
            self.tstp_var.tables[idx].insert(
                partitioned_values_list[idx],
                feature_status,
            )
        update_ops.append(mht_update)

    return control_flow_ops.group(update_ops)

  def filter(self, input_tensor=None, **kwargs):
    """
    Filter new feature keys by probability before training.
    Prevent unpopular features from affecting training.

    Args:
      **kwargs: keyword arguments, including
      input_tensor: SparseTensor or DenseTensor.
                    Feature keys need to filter.
      probability: float. Filtering new feature keys by
                   probability, and permitting old keys.

    Returns:
      Tensor that are filtered for training.
    """

    if input_tensor is None:
      raise KeyError("filter method expects parameter `input_tensor`.")
    elif isinstance(input_tensor, ops.Tensor):
      input_type = "DenseTensor"
      values = input_tensor
    elif isinstance(input_tensor, sparse_tensor.SparseTensor):
      input_type = "SparseTensor"
      values = input_tensor.values
      indices = input_tensor.indices
    else:
      raise TypeError("input_tensor must be " \
                "either a SparseTensor or dense Tensor.")

    if 'probability' in kwargs:
      probability = kwargs['probability']
    else:
      raise KeyError("filter method expects parameter `probability`.")
    if not isinstance(probability, float):
      raise TypeError("probability must be a float.")
    if probability < 0.0 or probability > 1.0:
      raise ValueError("probability value must be in [0.0, 1.0].")

    unique_values, value_idx = array_ops.unique(values)
    status_values = array_ops.reshape(unique_values, (-1,))
    partition_index = \
        self.var.partition_fn(status_values, self.var.shard_num)
    partitioned_values_list, partitioned_indices_list = \
        dynamic_embedding_ops._partition(status_values,
                                         partition_index,
                                         self.var.shard_num)

    fv_list = []
    for idx, dev in enumerate(self.tstp_var.devices):
      with ops.device(dev):
        feature_status = \
            self.tstp_var.tables[idx].lookup(
                partitioned_values_list[idx],
                dynamic_default_values=self.default_tstp,
            )

        sub_fv = array_ops.reshape(feature_status, (-1,))
        fv_list.append(sub_fv)

    total_fv = dynamic_embedding_ops._stitch(fv_list, partitioned_indices_list)
    total_fv = array_ops.gather(total_fv, value_idx)

    value_size = array_ops.size(values)
    old_prob = array_ops.ones(value_size)
    new_prob = array_ops.fill([value_size], probability)
    random_prob = random_ops.random_uniform([value_size], maxval=1.0)

    condition = math_ops.greater(total_fv, self.default_tstp)
    total_prob = array_ops.where(condition, old_prob, new_prob)

    total_mask = math_ops.greater_equal(total_prob, random_prob)
    filter_values = array_ops.boolean_mask(values, total_mask)

    if input_type == "DenseTensor":
      filter_tensor = filter_values
    elif input_type == "SparseTensor":
      filter_indices = array_ops.boolean_mask(indices, total_mask)
      filter_tensor = sparse_tensor.SparseTensor(
          indices=filter_indices,
          values=filter_values,
          dense_shape=input_tensor.dense_shape)

    return filter_tensor

  def restrict(self, **kwargs):
    """
    Restrict the status table size, eliminate the oldest
    feature keys, if the size of variable grow too large for
    threshold.

    Args:
      **kwargs: Keyword arguments, including
      threshold: int. The threshold for feature number
        in variable. When restrict method is called, the table
        size will be reduced to 'threshold'.
      factor: int,float,tf.int32,tf.int64,tf.float32.
        If the table size is greater than threshold * factor,
        restricting wiil be triggered.

    Returns:
      An operation to restrict the size of variable.
    """
    try:
      self.threshold = kwargs['threshold']
    except:
      raise KeyError("restrict method expects parameter `threshold`.")
    if not isinstance(self.threshold, int):
      raise TypeError("threshold must be an integer.")
    if self.threshold < 0:
      raise ValueError("threshold must be greater or equal to zero.")

    factor = kwargs.get('factor', 1.0)
    if isinstance(factor, ops.Tensor):
      if factor.dtype not in (dtypes.int32, dtypes.int64, dtypes.float32):
        raise TypeError(
            'factor expects int, float, tf.int32, tf.int64, or tf.float32')
      factor = math_ops.cast(factor, dtype=dtypes.float32)
    if not isinstance(factor, (int, float)):
      raise TypeError(
          'factor expects int, float, tf.int32, tf.int64, or tf.float32')

    cond_size = math_ops.cast(self.threshold, dtype=dtypes.float32) * factor
    cond_size = math_ops.cast(cond_size, dtype=dtypes.int64)
    condition = math_ops.greater(self.tstp_var.size(), cond_size)
    restrict_status_ops = list()
    no_ops = list()

    for idx, dev in enumerate(self.tstp_var.devices):
      with ops.device(dev):
        sub_tk, sub_tv = self.tstp_var.tables[idx].export()
        sharded_threshold = int(self.threshold / self.tstp_var.shard_num)

        sub_tv = array_ops.reshape(sub_tv, (-1,))
        first_dim = array_ops.shape(sub_tv)[0]

        k_on_top = math_ops.cast(first_dim - sharded_threshold,
                                 dtype=dtypes.int32)
        k_on_top = math_ops.maximum(k_on_top, 0)
        _, removed_keys_ids = nn_ops.top_k(-sub_tv, k_on_top, sorted=False)
        removed_keys = array_ops.gather(sub_tk, removed_keys_ids)
        restrict_status_ops.append(
            self.tstp_var.tables[idx].remove(removed_keys))
        no_ops.append(control_flow_ops.no_op())
    restrict_op = control_flow_ops.cond(condition,
                                        lambda:restrict_status_ops,
                                        lambda:no_ops)

    return restrict_op


@tf_export("dynamic_embedding.FeatureFilter")
class FeatureFilter(object):
  """
  A feature_filter for constraining the variables sparse feature number,
  with keeping recording and eliminating the obsolete feature keys.
  Notice: FrequencyFilterPolicy running order:   update->filter->train
            1.update feature keys frequency
            2.filter feature keys by frequency
            3.train with filtering feature keys
          ProbabilityFilterPolicy running order: filter->update->train
            1.filter feature keys by probability
            2.update with filtering feature keys
            3.trian with filtering feature keys
  # Example:

  ```python
  # Get a FeatureFilter.
  feature_filter = tf.dynamic_embedding.FeatureFilter(
                   var_list=var_list,
                   policy=FrequencyFilterPolicy,
               )

  # Call update to get an operation to update policy status,
  # record feature keys status.
  # There is no need to call update in inference.
  update_op = feature_filter.update(input_tensor_list=tensor_list)

  # Call filter to get qualified feature keys for training.
  # There is no need to call filter in inference.
  threshold = 10
  filter_tensor_list = feature_filter.filter(frequency_threshold=threshold,
                                             input_tensor_list=tensor_list)
  use_filter = mode != PREDICT and
                   math_ops.equal(math_ops.mod(global_step, 100), 0)
  cur_tensor_list = tf.cond(use_filter,
                            lambda:filter_tensor_list,
                            lambda:tensor_list)
  
  # Call restrict to get an operation to restrict policy status,
  # limit the status table size.
  # There is no need to call restrict in inference.
  restrict_op = feature_filter.restrict(threshold=size)
  
  # Training with filtering keys
  # Call the minimize to the loss with optimizer.
  test_var, _ = tf.dynamic_embedding.embedding_lookup_sparse(
      embeddings,
      cur_tensor_list[idx],
      sp_weights=None,
      combiner="sum",
      return_trainable=True)
  pred = math_ops.matmul(test_var, x)
  loss = pred * pred

  with tf.control_dependencies(update_op):
    train_op = opt.minimize(loss)

  with tf.Session() as sess:
    ...

    for step in range(num_iter):
      ...
      #Traning with filter keys
      #Need close 'update', 'filter' and 'restrict' in inference
      sess.run(train_op)
      if step % 1000 == 0:
        sess.run(restrict_op)
      ...

    ...
  ```

  """

  def __init__(self,
               var_list=None,
               default_value_list=None,
               policy=FrequencyFilterPolicy):
    """
    Creates a `FeatureFilter` object. Each variable in var_list
    of the same FeatureFilter instance share the same policy.

    Args:
      var_list: A list of `tf.dynamic_embedding.Variable` objects.
      default_value_list: A list of 'int' for default_value initializing.
        Some policies may use this for initializing status.
      policy: A FilterPolicy class to specify the rules for
        recoding, updating, and filtering the variable status in var_list.
    """
    if not issubclass(policy, FilterPolicy):
      raise TypeError("policy must be subclass of" \
                      "FilterPolicy object.")

    if var_list in [None, []]:
      raise ValueError("var_list must have a variable at least.")
    if default_value_list is not None:
      if len(default_value_list) != len(var_list):
        raise ValueError("default_value_list length" \
                         "must be equal to var_list.")
    else:
      default_value_list = len(var_list) * [0]

    self.var_list = var_list
    self.policy_list = []

    for idx, var in enumerate(self.var_list):
      self.policy_list.append(policy(var, default_value=default_value_list[idx]))

  def update(self, input_tensor_list=None, **kwargs):
    """
    Update the status for every variable in var_list.
    Each variable processes different sparse tensor keys.

    Args:
      input_tensor_list: A list of `Tensor` objects.
        For each variable, a sparse tensor should be passed to
        the FilterPolicy to update method according to the index.
      **kwargs: Optional keyword arguments to be passed to
        the FilterPolicy update method.

    Returns:
      A list of operations to update the status for every variable.
    """
    update_ops = []

    if input_tensor_list is None:
      raise KeyError("update method expects parameter" \
                     "`input_tensor_list`.")
    elif not isinstance(input_tensor_list, list):
      raise TypeError("input_tensor_list must be a list.")
    elif len(input_tensor_list) != len(self.var_list):
      raise ValueError("input_tensor_list length" \
                       "must be equal to var_list length.")

    for idx, policy in enumerate(self.policy_list):
      update_ops.append(
          policy.update(input_tensor=input_tensor_list[idx], **kwargs))

    return update_ops

  def filter(self, input_tensor_list=None, **kwargs):
    """
    Filter keys for every variable in var_list.
    Each variable processes different sparse tensor keys.

    Args:
      input_tensor_list: A list of `Tensor` objects.
        For each variable, a sparse tensor should be passed
        the FilterPolicy to update method according to the index.
      **kwargs: Optional keyword arguments to be passed to
        the FilterPolicy filter method.

    Returns:
      Tensor list that filter for training
    """
    filter_list = []

    if input_tensor_list is None:
      raise KeyError("update method expects parameter" \
                     "`input_tensor_list`.")
    elif not isinstance(input_tensor_list, list):
      raise TypeError("input_tensor_list must be a list.")
    elif len(input_tensor_list) != len(self.var_list):
      raise ValueError("input_tensor_list length" \
                       "must be equal to var_list length.")

    for idx, policy in enumerate(self.policy_list):
      filter_list.append(
          policy.filter(input_tensor=input_tensor_list[idx], **kwargs))

    return filter_list

  def restrict(self, **kwargs):
    """
    Restrict the variables for every variable in var_list.

    Args:
      **kwargs: Optional keyword arguments passed to the
        method policy.restrict(**kwargs). For example,
        in the `restrict` method of `FilterFrequencyPolicy`
        has parameters `threshold` and `factor`.

    Returns:
      A list of operation to restrict variables.
    """
    restrict_op = []
    for policy in self.policy_list:
      restrict_op.append(policy.restrict(**kwargs))
    return restrict_op

