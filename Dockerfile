#FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER nejedly@isibrno.cz
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.
#RUN apt-get update && apt-get upgrade -y && apt-get clean
#RUN apt-get install -y python3.7 python3.7-dev python3.7-distutils python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt


