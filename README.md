# labs-nn

## Installation

1. Clone repository

2. Build Docker image
```shell script
docker build -t labs-nn/tensorflow:2.3.0-gpu-jupyter .
```
or build with docker run
```shell script
./run.sh -b
```

3. Run docker image
```
docker run --gpus all --rm -p 8888:8888 -p 6006:6006 -v $(pwd):/project -it --name labs-nn-project labs-nn/tensorflow:2.3.0-gpu-jupyter
```
or
```
./run.sh
```
