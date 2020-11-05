FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

RUN mkdir project

WORKDIR /project/

COPY ./requirements.txt /project/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root