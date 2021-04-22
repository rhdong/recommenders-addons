#! /bin/bash
#
# Copyright 2021 the deepx authors.
# Author: Yafei Zhang (zhangyafeikimi@gmail.com)
#

cd $(dirname $0)

function sync() {
    echo Syncing $1 to $2...
    mkdir -p $(dirname $2)
    cp $1 $2
}

SYNC_FILES="\
include/deepx_core/common/hash_map.h \
include/deepx_core/tensor/ll_math.h"

set -e
git clone https://github.com/Tencent/deepx_core.git
for file in $SYNC_FILES; do
sync deepx_core/$file $file
done
rm -rf deepx_core
