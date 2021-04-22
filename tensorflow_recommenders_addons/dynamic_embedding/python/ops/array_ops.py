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
"""TFRA optimized version operations of tensorflow array_ops."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow_recommenders_addons.utils.resource_loader import LazySO

gen_array_ops = LazySO("dynamic_embedding/core/_gen_array_ops.so").ops


def unique(x, out_idx=dtypes.int32, name=None):
  """The TFRA optimzed tf.unique.
  """
  return gen_array_ops.tfra_unique(x, out_idx, name)


def unique_v2(x, axis, out_idx=dtypes.int32, name=None):
  """The TFRA optimzed tf.unique_v2.
  """
  return gen_array_ops.tfra_unique_v2(x, axis, out_idx, name)


def unique_with_counts(x, out_idx=dtypes.int32, name=None):
  """The TFRA optimzed tf.unique_with_counts.
  """
  return gen_array_ops.tfra_unique_with_counts(x, out_idx, name)


def unique_with_counts_v2(x, axis, out_idx=dtypes.int32, name=None):
  """The TFRA optimzed tf.unique_with_counts_v2.
  """
  return gen_array_ops.tfra_unique_with_counts_v2(x, axis, out_idx, name)
