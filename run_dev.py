from models.ReGroupNet import resnet18, BasicBlock
import torch

from datasets.datasets import get_tiny_imagenet_wds
from bnn import BConfig, prepare_binary_model, Identity
from run_experiment import run_experiment

from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    BasicScaleBinarizer,
    InputBiasBinarizer
    
)

model = resnet18(num_classes=200, rprelu=False)

#model = models.__dict__["resnet18"](stem_type="basic", num_classes=200)

bconfig = BConfig(
    activation_pre_process=InputBiasBinarizer,
    activation_post_process=BasicScaleBinarizer,
    weight_pre_process=XNORWeightBinarizer.with_args(compute_alpha=False,
                                                     center_weights=True),
)

model = prepare_binary_model(
    model,
    bconfig,
    custom_config_layers_name={
        "first_conv": BConfig(),
        # "fc": BConfig(),
    },
)


TARGET = "label"

run_experiment(model, get_tiny_imagenet_wds, TARGET)
