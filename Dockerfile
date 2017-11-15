FROM tensorflow/tensorflow:1.3.0-devel-gpu-py3

# Download and build Keras
RUN pwd && ls &&  git clone git://github.com/fchollet/keras.git &&\
    cd keras && \
    python3 setup.py install

# Download and unzip dataset
RUN pwd && ls && wget http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip && \
    unzip sushi_or_sandwich_photos.zip

# Configure the dataset
ADD split_data.py /root
RUN pwd && ls && python3 split_data.py

# Run training and validation codes
ADD train.py /root
ADD dltools /root/dltools
WORKDIR /root
ENV PYTHONPATH /root
RUN apt-get update && apt-get install -y --no-install-recommends python3-tk && pwd && ls && cd dltools && pwd && ls

RUN CUDA_VISIBLE_DEVICES="-1" python3 train.py
