FROM tensorflow/tensorflow:1.3.0-devel-gpu

# Download and build Keras
RUN git clone git://github.com/fchollet/keras.git &&\
    cd keras && \
    python setup.py install && \
    cd ..

# Download the repository
RUN pwd && ls && git clone https://github.com/ub216/cook && cd cook && ls

# Download and split the dataset
RUN wget http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip && \
    unzip sushi_or_sandwich_photos.zip && \
    pwd && ls && \
    python split_data.py

# Run training and validation codes
RUN python train.py
