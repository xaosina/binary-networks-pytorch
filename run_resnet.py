from local_models.resnetsCIFAR import ResNet18
import torch

from local_datasets.dataset_makers import get_cifar100
from bnn import BConfig, prepare_binary_model, Identity
from run_experiment import run_experiment

from bnn.ops import (
    BasicBinarizer,
    LearnableScale,
    LearnableBiasScale,
    InputBiasBinarizer,
    XNORWeightBinarizer,
)

model = ResNet18()

# model = models.__dict__["resnet18"](stem_type="basic", num_classes=200)

# bconfig = BConfig(
#     activation_pre_process=BasicBinarizer,
#     activation_post_process=LearnableScale,
#     weight_pre_process=XNORWeightBinarizer.with_args(compute_alpha=False,
#                                                      center_weights=True),
# )


bconfig = BConfig(
    activation_pre_process=BasicBinarizer,
    activation_post_process=Identity,
    weight_pre_process=BasicBinarizer,
)

model = prepare_binary_model(
    model,
    bconfig,
    custom_config_layers_name={
        "first_conv": BConfig(),
        "linear": BConfig(),
        # "fc": BConfig(),
    },
)


TARGET = "label"

run_experiment(model, get_cifar100, TARGET)
