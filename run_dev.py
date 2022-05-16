from models.group_net import resnet18
from models.reactnet import reactnet
import bnn.models.resnet as models

from datasets.datasets import get_tiny_image_net
from bnn import BConfig, prepare_binary_model, Identity
from run_experiment import run_experiment

from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    AdvancedNoisyInputBinarizer,
    BasicScaleBinarizer,
    BiasPostprocess,
    BiasInputBinarizer
    
)

#model = resnet18(num_classes=200)

model = models.__dict__["resnet18"](stem_type="basic", num_classes=200)

bconfig = BConfig(
    activation_pre_process=BasicInputBinarizer,
    activation_post_process=BasicScaleBinarizer,
    weight_pre_process=XNORWeightBinarizer.with_args(compute_alpha=False,
                                                     center_weights=True),
)

model = prepare_binary_model(
    model,
    bconfig,
    custom_config_layers_name={
        "first_conv": BConfig(),
        #     "move1_a": BConfig(),
        #     "move2_c": BConfig(),
        #     "move2_a": BConfig(),
    },
)

#print(model)

TARGET = "label"

run_experiment(model, get_tiny_image_net, TARGET)

# np.save(
#     "./filters_TanhBinarizer_L",
#     model.conv1.weight.data.detach().cpu().numpy(),
# )