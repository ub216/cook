FROM ubuntu:16.04

RUN pwd && ls

ADD train.py /

RUN pwd && ls
