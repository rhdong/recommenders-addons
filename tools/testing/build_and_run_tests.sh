#!/usr/bin/env bash
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
#
# ==============================================================================
# usage: bash tools/testing/build_and_run_tests.sh

set -x -e

export CC_OPT_FLAGS='-mavx'

python -m pip install -r tools/install_deps/pytest.txt -e ./
TF_NEED_CUDA=$TF_NEED_CUDA python ./configure.py
bash tools/install_so_files.sh


EXTRA_ARGS="-n 10"


python -m pytest -v -s --functions-durations=20 --modules-durations=5 $EXTRA_ARGS ./tensorflow_recommenders_addons
