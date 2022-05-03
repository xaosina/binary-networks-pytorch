import numpy as np

import torch.nn as nn
import bnn.models.resnet as models
from datasets.datasets import get_imagenet_wds, get_tiny_imagenet_wds
from bnn import BConfig, prepare_binary_model, Identity
from run_experiment import run_experiment


from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    TanhBinarizer,
    NoisyTanhBinarizer,
    BasicScaleBinarizer,
)

model = models.__dict__["resnet18"](stem_type="basic", num_classes=200, block_type=models.PreBasicBlock, activation=nn.PReLU)

bconfig = BConfig(
    activation_pre_process=BasicInputBinarizer,
    activation_post_process=BasicScaleBinarizer,
    weight_pre_process=XNORWeightBinarizer.with_args(compute_alpha=True, center_weights=True),
)


# model = prepare_binary_model(
#     model,
#     bconfig,
#     custom_config_layers_name={
#         "_last_": BConfig(),
#         "_first_": BConfig(),
#     },
# )
target = "label"
run_experiment(model, get_tiny_imagenet_wds, target)