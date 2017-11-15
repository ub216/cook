FROM tensorflow/tensorflow:1.3.0-devel-gpu

ADD dltools /root/dltools
RUN pwd && ls && cd dltools && pwd && ls
