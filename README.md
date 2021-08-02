# GPT2 AI Data Transformer 
## Fine tuning for Q&A task

![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)

Open AI model, we export features from a database by finetuning the huggingface model

- Dataset Greek ID Cards
- Multilingual Model 
- ✨DNN✨

## Features

- First 
- Second


## Docker

Create a folder on your local machine e.g. and clone the repository
```
mkdir project
cd project

git clone https://github.com/Erjon-19/GPT2_AI_Data_Trasnformer.git
```
### Build container
Build image

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
 docker run --gpus 2 -v $(cd -P .. && pwd -P)/model_save/:/root/model_save/ -v $(pwd)/:/root/c1/ -ti my_c1 /bin/bash -c "cd /root/c1 && source activate ml && python3 gpt2_training.py"
 ```

### Run Demo
**GPU**
```sh
docker run --gpus 1 -v $(cd -P .. && pwd -P)/model_save/:/root/model_save/ -v $(pwd)/:/root/c1/ -ti my_c1 /bin/bash -c "cd /root/c1 && source activate ml && python3 Demo.py"
```
**CPU**
```sh
docker run -v $(cd -P .. && pwd -P)/model_save/:/root/model_save/ -v $(pwd)/:/root/c1/ -ti my_c1 /bin/bash -c "cd /root/c1 && source activate ml && python3 Demo.py"
```

## License
