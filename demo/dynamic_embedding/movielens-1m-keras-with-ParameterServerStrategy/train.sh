#!/usr/bin/env bash

sh stop.sh

python movielens-1m-keras-with-ps-strategy.py --ps="localhost:2220,localhost:2221" --worker="localhost:2230,localhost:2231" --task_type="ps" --task_index=0

python movielens-1m-keras-with-ps-strategy.py --ps="localhost:2220,localhost:2221" --worker="localhost:2230,localhost:2231" --task_type="ps" --task_index=1

python movielens-1m-keras-with-ps-strategy.py --ps="localhost:2220,localhost:2221" --worker="localhost:2230,localhost:2231" --task_type="worker" --task_index=0

python movielens-1m-keras-with-ps-strategy.py --ps="localhost:2220,localhost:2221" --worker="localhost:2230,localhost:2231" --task_type="worker" --task_index=1