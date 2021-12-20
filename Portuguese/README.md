# GPT-2 AI Data Transformer (GLASS H2020 project)
## Fine tuning GPT2 for Q&A task

![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)

Open AI model, we export features from a database by finetuning the huggingface model GPT-2

- Dataset Portuguese IDs
- Multilingual Model 
- ✨DNN✨
- Deploy

## Intro
In this repo we finetuned the small version of gpt2 model with 117 of parameteres for Q.A. task, so the the input to the model is an important parameter. 

## Features

- Question Answering Tast 
- Finetuning GPT-2 with any Dataset
- Modular finetuning can run either on GPU or CPU
- Deploy with Docker and Gradio
- For evaluation metrics we use f1-score
- Adapt to any new dataset for Q.A. task

## Pre-Requisites
- **NVIDIA Drivers**
- **Ubuntu 20.04 or 18.04**
- If drivers not installed copy the following commands

```sh
sudo ubuntu-drivers autoinstall

sudo reboot
```
# Setting up Docker & Nvidia runtime
Follow [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) link if you have other Linux distro 

Docker-CE on Ubuntu can be setup using Docker’s official convenience script:

```sh
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```
### NVIDIA Container Toolkit
```sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
sudo apt-get update
   
sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker
```
The working setup can be tested by running a base CUDA container

```sh
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```


## Build & Run container

Create a folder on your local machine e.g. and clone the repository
```sh
mkdir gpt2QA
cd gpt2QA
```

```sh
git clone https://github.com/Erjon-19/Transformer_GPT2QA_Portuguese_ids.git
```
### Build container
First change directory and then Build the container

```sh
docker build -f dockerfile -t my_c1 .
```
### Run training 
In order to run training we must mount two folders
1. The current folder where we have the files so if we make any changes to them there is NO NEED to build up again the container   
2. One folder up for saving the fine tuned model
3. Also in run commant we specify the number of gpus 

**GPU**
 ```sh 
 docker run --gpus all -v $(cd -P .. && pwd -P)/model_save/:/root/model_save/ -v $(pwd)/:/root/c1/ -ti my_c1 /bin/bash -c "cd /root/c1 && source activate ml && python3 gpt2_training_Porto.py"
 ```

### Run Demo
**GPU**
```sh
docker run --gpus 1 -v $(cd -P .. && pwd -P)/model_save/:/root/model_save/ -v $(pwd)/:/root/c1/ -ti my_c1 /bin/bash -c "cd /root/c1 && source activate ml && python3 Demo_Porto.py"
```
**CPU**
```sh
docker run -v $(cd -P .. && pwd -P)/model_save/:/root/model_save/ -v $(pwd)/:/root/c1/ -ti my_c1 /bin/bash -c "cd /root/c1 && source activate ml && python3 Demo_Porto.py"
```

### Metrics
**F1 score**
```sh
docker run -v $(cd -P .. && pwd -P)/model_save/:/root/model_save/ -v $(pwd)/:/root/c1/ -ti my_c1 /bin/bash -c "cd /root/c1 && source activate ml && python3 test_f1_score.py"
```
## License
