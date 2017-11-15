FROM tensorflow/tensorflow:1.3.0-devel-gpu

# Download and build Keras
RUN pwd && ls &&  git clone git://github.com/fchollet/keras.git &&\
    cd keras && \
    python setup.py install

# Download and unzip dataset
RUN pwd && ls && wget http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip && \
    unzip sushi_or_sandwich_photos.zip

# Configure the dataset
ADD split_data.py /root
RUN pwd && ls && python split_data.py

# Run training and validation codes
ADD train.py /root
ADD dltools /root/dltools
WORKDIR /root
ENV PYTHONPATH /root/dltools
RUN pwd && ls && cd dltools && pwd && ls

RUN python train.py
