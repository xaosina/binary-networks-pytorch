<hr>

  
# Binary Neural Networks (BNN)

  
## Installation

### Requirements

* Python 3.7+
* iopath
* PyTorch (>=1.8)

#### Optional:
- webdataset


## Quick start
### 1. Choose your configuration file
Path: `configs/env_{dima|egor}.yaml`
Specify:
1. PROJECT.ROOT - current dir location
2. HARDWARE.WORKERS - num of workers
3. TRAINING:
	  EPOCHS
	  BATCH_SIZE
4. EXPERIMENT.DIR - Logging directory

The rest can be skipped.

#### 2. Filling `run_dev.py`
You can find `run_dev.py` examples in `recipes` folder.
Basically in `run_dev` you prepare **model** and **dataset** to pass it to the trainer.

Model examples for ImageNet can be found in `models`.
Model examples for Cifar can be found in `models/cifar`.

Datasets can be imported from `datasets/datasets.py`
**NOTE: In order to use them you have to specify data pathes manually inside `datasets.py`**
Some datasets require `webdataset` package.

##### Example of preparing model:
```python
from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    BasicScaleBinarizer,
    InputBiasBinarizer # corresponds to RSign from ReActNet 
) # Binarizers for pre/post processing of actiations and weights.

model = resnet18(num_classes=1000, rprelu=False) # model, imported from `/models`
  
bconfig = BConfig(
    activation_pre_process=InputBiasBinarizer, # corresponds to RSign from ReActNet
    activation_post_process=BasicScaleBinarizer,
    weight_pre_process=XNORWeightBinarizer.with_args(compute_alpha=False,
                                                     center_weights=True),
) # Binarization config. 

model = prepare_binary_model(
    model,
    bconfig,
    custom_config_layers_name={
        "first_conv": BConfig(),
    },
) # Transform regular DNN into BNN.
```

#### 3. Training done by running `run_dev.py`
```
usage: run_dev.py [-h] [--exp_name exp_name] [--gpu gpu] -u {dima,egor} [--seed seed]

PyTorch Binary Network Training

optional arguments:
  -h, --help            show this help message and exit
  --exp_name exp_name   experiment name
  --gpu gpu             gpu
  -u {dima,egor}, --user {dima,egor} # Which config to use
  --seed seed           seed
```

---
## Binarizers guide

### **1. Explicit usage**

Similarly with the pytorch quantization module we can define a binarization configuration  that will contains the binarization strategies(modules) used. Once defined, the `prepare_binary_model` function will propagate them to all nodes and then swap the modules with the fake binarized ones.

Alternatively, the user can define manually, at network creation time, the bconfig for each layer and then call then `convert` function to swap the modules appropriately.

```python
import torch
import torchvision.models as models

from bnn import BConfig, prepare_binary_model
# Import a few examples of quantizers
from bnn.ops import BasicInputBinarizer, BasicScaleBinarizer, XNORWeightBinarizer

# Create your desire model (note the default R18 may be suboptimal)
# additional binarization friendly models are available in bnn.models
model = models.resnet18(pretrained=False)

# Define the binarization configuration and assign it to the model
bconfig = BConfig(
    activation_pre_process = BasicInputBinarizer,
    activation_post_process = BasicScaleBinarizer,
    # optionally, one can pass certain custom variables
    weight_pre_process = XNORWeightBinarizer.with_args(center_weights=True)
)
# Convert the model appropiately, propagating the changes from parent node to leafs
# The custom_config_layers_name syntax will perform a match based on the layer name, setting a custom quantization function.
bmodel = prepare_binary_model(model, bconfig, custom_config_layers_name=['conv1' : BConfig()])

# You can also ignore certain layers using the ignore_layers_name.
# To pass regex expression, frame them between $ symbols, i.e.: $expression$.

  

```
  
### **2. Implementing a custom weight binarizer**

Implementing custom operations is a straightforward process. You can simply define your new classpython register class to a given module:  

```python
import torch.nn as nn
import torch.nn.functional as F

class CustomOutputBinarizer(nn.Module):
    def __init__(self):
        super(CustomOutputBinarizer, self).__init__()
    def forward(self, x_after, x_before):
        # scale binarizer takes a list of input containg [conv_output and conv_input]
        return F.normalize(x_after, p=2) # operate on the conv_output

class CustomInputBinarizer(nn.Module):
    def __init__(self):
        super(CustomInputBinarizer, self).__init__()
    def forward(self, x):
        # dummy example of using sign instead of tanh
        return torch.tanh(x) # operate on the conv_output

# apply the custom functions into the binarization model

bconfig = BConfig(
    activation_pre_process = CustomInputBinarizer,
    activation_post_process = CustomOutputBinarizer,
    weight_pre_process = nn.Identity # this will keep the weights real
)
```