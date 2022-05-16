import copy

from models.group_net import resnet18
from models.reactnet import reactnet
import bnn.models.resnet as models

from datasets.datasets import get_tiny_image_net
from run_experiment import validate

from estimate_flops import get_stats
from ensembler import Ensmembler

from bnn import BConfig, prepare_binary_model, Identity

from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    BasicScaleBinarizer,
    BiasPostprocess,
    BiasInputBinarizer
    
)


bconfig = BConfig(
    activation_pre_process=BasicInputBinarizer,
    activation_post_process=BasicScaleBinarizer,
    weight_pre_process=XNORWeightBinarizer.with_args(compute_alpha=False, center_weights=True),
)

model = models.__dict__["resnet18"](stem_type="basic", num_classes=200)
get_stats(model)

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



models_list = [(copy.deepcopy(model),'/home/dev/data_main/LOGS/BNN/binary_exp_time_env/resnet18v2_binirized/model_best.pth.tar'),
          (copy.deepcopy(model),
           '/home/dev/data_main/LOGS/BNN/binary_exp_time_env/resnnet_binary_baisc/model_best.pth.tar'),
          (copy.deepcopy(model),
           '/home/dev/data_main/LOGS/BNN/binary_exp_time_env/resnet18_v4_binarized/model_best.pth.tar'),
          ]


model = Ensmembler(models_list, mix_type='voting')

TARGET = "label"

validate(model, get_tiny_image_net, TARGET)

# np.save(
#     "./filters_TanhBinarizer_L",
#     model.conv1.weight.data.detach().cpu().numpy(),
# )