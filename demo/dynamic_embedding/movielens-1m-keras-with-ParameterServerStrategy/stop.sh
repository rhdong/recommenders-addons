ps -ef|grep "movielens-1m-keras-with-ps-strategy.py"|grep -v grep|awk '{print $2}'| xargs kill -9
sleep 1
