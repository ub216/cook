FROM tensorflow/tensorflow:1.2.0-devel-gpu-py3

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN apt-get update && apt-get install -y --no-install-recommends python3-tk

# Download and build Keras
RUN pwd && ls &&  wget http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip \
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

RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH python3 train.py
