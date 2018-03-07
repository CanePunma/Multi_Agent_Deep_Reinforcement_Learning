# Multi Agent Deep Reinforcement Learning For Autonomous Vehicle Relocation


## Summary

Autonomous vehicles are becoming more common in city transportation. Companies will begin to find a need to teach these vehicles smart city fleet coordination. Currently, simulation based modeling along with hand coded rules dictate the decision making of these autonomous vehicles. We believe that complex intelligent behavior can be learned by these agents through Reinforcement Learning. 

## Model Description

In this repo, we discuss our work for solving this system by adapting the Deep Q-Learning (DQN) model to the multi-agent setting. Our approach applies deep reinforcement learning by combining convolutional neural networks with DQN to teach agents to fulfill customer demand in an environment that is partially observable to them. 

For more details, the draft of the white paper can be found in the repo.

## Downloading & Using the API

0. Download Docker. If you are on Mac or Windows, Docker Compose will be automatically installed. On Linux, make sure you have the latest version of Compose. If you're using Docker for Windows on Windows 10 pro or later, you must also switch to Linux containers.


Install the Community Edition for Ubuntu __[here](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)__.

If you are using a machine that has GPUs enabled install the nvidia docker __[here](https://github.com/NVIDIA/nvidia-docker)__.

This requires CUDA Toolkit Version 8+ and CuDnn 5+, more info can be found __[here](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA)__.

1. Navigate to the root folder where the Dockerfile exists and build the docker image using the following Docker command:

    `docker build -t [NAME]:[VERSION] .`

2. Start the API using the following Docker command:

    `{ docker run --runtime=nvidia -p [5001]:[5001] -i -t [NAME]:[VERSION] }`
    
    Use the parameter --runtime=nvidia if you are running the Nvidia Docker

3. Open the user interface in the browser by going to __[http://localhost:5001/initialize](http://localhost:5001/)__

4. Open up the index.html file (Will be hosting the backend as an api on a aws route, currently just using localhost)

If you can't seem to get Docker working on your local system, you can try the following steps.

1. Download Anaconda with Python 2.7 __[here](https://conda.io/docs/user-guide/install/download.html)__.
2. Conda Install: (numpy, Flask, h5py, keras, theano)
3. cd app, and Run python main.py

The model should be able to run on your local machines CPU.

## Contributors

canepunma1@gmail.com

