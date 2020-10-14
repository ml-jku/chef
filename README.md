# CHEF
[Cross-domain few-shot learning by representation fusion](https://arxiv.org/abs/2010.06498)

# Download and prepare data

## miniImagenet

Download the miniImagenet data from 
[here](https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view).
We use the data split proposed by 
[Ravi & Larochelle](https://openreview.net/pdf?id=rJY0-Kcll). 
Create the folders `images_train`, `images_val`, `images_test` and place 
the respective files in them, as well as a folder `images_trainval` that 
must contain all images from `images_train` and `images_val`. 

## tieredImagenet

Download the tiereImagenet data from 
[here](https://datasets.d2.mpi-inf.mpg.de/yaoyaoliu/tiered_imagenet.tar). 
Extract it and create a folder `trainval` containing all images 
from the folders `train` and `val`. 

## Set up horizontal data splits

Run `python3 make_miniImagenet_hsplit.py` and 
`python3 make_tieredImagenet_hsplit.py` to set up the horizontal 
data splits for pretraining. 

## Cross-domain data

Follow the instructions in [this repo](https://github.com/IBM/cdfsl-benchmark) 
to acquire and set up the cross-domain data sets. 

# Pre-training

Run `python3 pretrain.py config/pretrain_{res10,res12,conv64}_{tier,mini}.json`.

# Testing

To test the pre-trained ResNet-10 on the four cross-domain data sets run 
`python3 xdom_res10.py config/xdom_res10.json --dataset {isic,cropdisease,eurosat,chest}`. 
To test the Imagenet-pre-trained ResNet-18 from PyTorch on the four 
cross-domain data sets run 
`python3 xdom_res10.py config/xdom_res18.json --dataset {isic,cropdisease,eurosat,chest}`. 
Without the `--dataset` option, the model will be run on the miniImagenet test 
set by default. 

To test the pre-trained ResNet-12 and Conv-4 on the miniImagenet and 
tieredImagenet test sets run 
`python3 test.py config/{res12,conv64}_{mini,tier}_{1,5}shot.json`. 

