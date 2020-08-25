#!/usr/bin/env bash

if [ "$1" = "-b" ]
then
  docker build -t labs-nn/tensorflow:2.3.0-gpu-jupyter .
fi

docker run --gpus all --rm -p 8888:8888 -p 6006:6006 -v $(pwd):/project -it --name labs-nn-project labs-nn/tensorflow:2.3.0-gpu-jupyter