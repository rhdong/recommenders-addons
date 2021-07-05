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
"""unit tests of redis table ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))


class RedisTableTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_dynamic_embedding_variable_set_init_size(self):
    with self.session(use_gpu=False, config=default_config):
      redis_config = de.RedisTableConfig(connection_mode=1,
                                         master_name="master",
                                         host_ip="127.0.0.1",
                                         host_port=26379,
                                         password="",
                                         db=0,
                                         connect_timeout=1000)
      table = de.get_variable(
          "2021-" + str(id),
          dtypes.int64,
          dtypes.int32,
          initializer=0,
          dim=8,
          kv_creator=de.RedisTableCreator(config=redis_config))
      self.evaluate(table.size())


if __name__ == "__main__":
  test.main()
