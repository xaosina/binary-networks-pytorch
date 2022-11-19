from models.DeepLabV3 import resnet50, Bottleneck
import torch

from datasets.datasets import get_pascal
from bnn import BConfig, prepare_binary_model, Identity
from run_experiment import run_experiment

from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    BasicScaleBinarizer,
    InputBiasBinarizer

)

model = resnet50(
    pretrained=True,
    num_classes=21,
    num_groups=None,
    weight_std=False,
    beta=False,
)

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
        "conv1": BConfig(),
        # "fc": BConfig(),
    },
)


TARGET = "label"

run_experiment(model, get_pascal, TARGET)
