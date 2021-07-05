# Copyright 2020 The TensorFlow Recommenders-Addons Authors.
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

# lint-as: python3

from tensorflow_recommenders_addons import dynamic_embedding as de


class KVCreator(object):

  def __init__(self, config=None):
    self.config = config

  def create(self,
             key_dtype=None,
             value_dtype=None,
             default_value=None,
             name=None,
             checkpoint=None,
             init_size=None,
             config=None):

    raise NotImplementedError('create function must be implemented')


class CuckooHashTableConfig(object):

  def __init__(self):
    """ CuckooHashTableConfig include nothing for parameter default satisfied.
    """
    pass


class CuckooHashTableCreator(KVCreator):

  def create(
      self,
      key_dtype=None,
      value_dtype=None,
      default_value=None,
      name=None,
      checkpoint=None,
      init_size=None,
      config=None,
  ):
    return de.CuckooHashTable(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        default_value=default_value,
        name=name,
        checkpoint=checkpoint,
        init_size=init_size,
        config=config,
    )


class RedisTableConfig(object):

  def __init__(
      self,
      connection_mode=1,
      master_name="master",
      host_ip="127.0.0.1",
      host_port=26379,
      password="",
      db=0,
      connect_timeout=1000,  # milliseconds
      socket_timeout=1000,  # milliseconds
      pool_size=20,
      wait_timeout=100000000,  # milliseconds
      connection_lifetime=100,  # minutes
      sentinel_connect_timeout=1000,  # milliseconds
      sentinel_socket_timeout=1000,  # milliseconds
      storage_slice=1,  # For deciding hash tag, which usually is how many Redis instance may be used in the trainning.
      using_MD5_prefix_name=False,  # 1=true, 0=false
      model_tag="test",  #  model_tag for version and any other information
      using_model_lib=True,
      model_lib_abs_dir="/tmp/",
  ):
    self.connection_mode = connection_mode
    self.master_name = master_name
    self.host_ip = host_ip
    self.host_port = host_port
    self.password = password
    self.db = db
    self.connect_timeout = connect_timeout
    self.socket_timeout = socket_timeout
    self.pool_size = pool_size
    self.wait_timeout = wait_timeout
    self.connection_lifetime = connection_lifetime
    self.sentinel_connect_timeout = sentinel_connect_timeout
    self.sentinel_socket_timeout = sentinel_socket_timeout
    self.storage_slice = storage_slice
    self.using_MD5_prefix_name = using_MD5_prefix_name
    self.model_tag = model_tag
    self.using_model_lib = using_model_lib
    self.model_lib_abs_dir = model_lib_abs_dir


class RedisTableCreator(KVCreator):

  def create(
      self,
      key_dtype=None,
      value_dtype=None,
      default_value=None,
      name=None,
      checkpoint=None,
      init_size=None,
      config=None,
  ):
    real_config = config if config is not None else self.config
    if not isinstance(real_config, RedisTableConfig):
      raise TypeError("config should be instance of 'config', but got ",
                      str(type(real_config)))
    return de.RedisTable(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        default_value=default_value,
        name=name,
        checkpoint=checkpoint,
        config=self.config,
    )
