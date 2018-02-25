
# Welcome to *PyTorchNet*!

****PyTorchNet**** is a Machine Learning framework that is built on top of [PyTorch](https://github.com/pytorch/pytorch). And, it uses [Visdom](https://github.com/facebookresearch/visdom) and [Plotly](https://github.com/plotly) for visualization.

PyTorchNet is easy to be customized by creating the necessary classes:
 1. **Data Loading**: a dataset class is required to load the data.
 2. **Model Design**: a nn.Module class that represents the network model.
 3. **Loss Method**: an appropriate class for the loss, for example CrossEntropyLoss or MSELoss.
 4. **Evaluation Metric**: a class to measure the accuracy of the results.

# Structure
PyTorchNet consists of the following packages:
## Datasets
This is for loading and transforming datasets.
## Models
Network models are kept in this package. It already includes [ResNet](https://arxiv.org/abs/1512.03385), [PreActResNet](https://arxiv.org/abs/1603.05027), [Stacked Hourglass](https://arxiv.org/abs/1603.06937) and [SphereFace](https://arxiv.org/abs/1704.08063).
## Losses
There are number of different choices available for Classification or Regression. New loss methods can be put here.
## Evaluates
There are number of different choices available for Classification or Regression. New accuracy metrics can be put here.
## Plugins
There are already three different plugins available:
1. **Monitor**:
2. **Logger**: 
3. **Visualizer**:
## Root
 - main
 - dataloader
 - train
 - test

# Setup
First, you need to download PyTorchNet by calling the following command:
> git clone --recursive https://github.com/human-analysis/pytorchnet.git

Since PyTorchNet relies on several Python packages, you need to make sure that the requirements exist by executing the following command in the *pytorchnet* directory:
> pip install -r requirements.txt

**Notice**
* If you do not have Pytorch or it does not meet the requirements, please follow the instruction on [the Pytorch website](http://pytorch.org/#pip-install-pytorch).

Congratulations!!! You are now ready to use PyTorchNet!

# Usage
Before running PyTorchNet, [Visdom](https://github.com/facebookresearch/visdom#usage) must be up and running. This can be done by:
> python -m visdom.server -p 8097

![screenshot from 2018-02-24 19-10-44](https://user-images.githubusercontent.com/24301047/36636554-1474092e-1997-11e8-8faa-cb4ee1fe9a85.png)

PyTorchNet comes with a classification example in which a [ResNet](https://arxiv.org/abs/1512.03385) model is trained for the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
> python [main.py](https://github.com/human-analysis/pytorchnet/blob/dev/main.py)

![screenshot from 2018-02-24 18-53-13](https://user-images.githubusercontent.com/24301047/36636539-abe73688-1996-11e8-83ea-c43318f24048.png)

![screenshot from 2018-02-24 18-58-03](https://user-images.githubusercontent.com/24301047/36636483-05f60038-1996-11e8-806e-895638396986.png)
# Configuration
PyTorchNet loads its parameters at the beginning via a config file and/or the command line.
## Config file
When PyTorchNet is being run, it will automatically load all parameters from [args.txt](https://github.com/human-analysis/pytorchnet/blob/master/args.txt) by default, if it exists. In order to load a custom config file, the following parameter can be used:
> python main.py --config custom_args.txt
### args.txt
> [Arguments]
>  
> port = 8097\
> env = main\
> same_env = Yes\
> log_type = traditional\
> save_results = No\
> \
> \# dataset options\
> dataroot = ./data\
> dataset_train = CIFAR10\
> dataset_test = CIFAR10\
> batch_size = 64


## Command line
Parameters can also be set in the command line when invoking [main.py](https://github.com/human-analysis/pytorchnet/blob/master/main.py). These parameters will precede the existing parameters in the configuration file.
> python [main.py](https://github.com/human-analysis/pytorchnet/blob/master/main.py) --log-type progressbar

