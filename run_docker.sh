#!/bin/sh

docker run \
--gpus all \
--volume $PWD:/src \
--volume /home/sysadmin/beemelmanns/tfrecords:/tfrecords \
--rm \
-it \
-u $(id -u):$(id -g) \
mononet_image \
python /src/network/train.py /src/configs/cityscapes_single.toml