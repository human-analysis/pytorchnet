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
Network models are kept in this package. It already includes [ResNet](https://arxiv.org/abs/1512.03385) and [Stacked Hourglass](https://arxiv.org/abs/1603.06937).
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

Since PyTorchNet relies on several Python packages, you need to install the requirements by executing the following command in the *pytorchnet* directory:
> pip install -r requirements.txt

Congratulations!!! You are now ready to use PyTorchNet!

# Usage
Before running PyTorchNet, [Visdom](https://github.com/facebookresearch/visdom#usage) must be up and running. This can be done by:
> python -m visdom.server -p 8097

PyTorchNet comes with a classification example in which a [ResNet](https://arxiv.org/abs/1512.03385) model is trained for the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
> python [main.py](https://github.com/human-analysis/pytorchnet/blob/dev/main.py)

![screenshot from 2018-02-24 18-53-13](https://user-images.githubusercontent.com/24301047/36636539-abe73688-1996-11e8-83ea-c43318f24048.png)

![screenshot from 2018-02-24 18-58-03](https://user-images.githubusercontent.com/24301047/36636483-05f60038-1996-11e8-806e-895638396986.png)

```{r, engine='bash', sample run}
python main.py --manual-seed 0 --dataset-train CIFAR10 --dataset-test CIFAR10 --dataroot ../ --nthreads 40 --optim-method Adam --batch-size 64 --learning-rate 3e-4 --beta1 0.9 --beta2 0.999 --nclasses 10 --nchannels 3 --resolution-high 32 --resolution-wide 32 --nepochs 100 --momentum 0.9 --weight-decay 0.0 --port 8097 --net-type resnet18 --nfilters 64 --cuda True --ngpu 1
```
