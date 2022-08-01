from models.ConvExpertNet import resnet18, BasicBlock
from models.layers.ebconv import EBConv2d
import torch
import torch.nn as nn
import bnn.layers as layers

from datasets.datasets import get_tiny_imagenet_wds
from bnn import BConfig, prepare_binary_model, Identity
from run_experiment import run_experiment

from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    BasicScaleBinarizer,
    InputBiasBinarizer
    
)

model = resnet18(num_classes=200, rprelu=False, use_only_first=False)

bconfig = BConfig(
    activation_pre_process=InputBiasBinarizer,
    activation_post_process=BasicScaleBinarizer,
    weight_pre_process=XNORWeightBinarizer.with_args(compute_alpha=False,
                                                     center_weights=True),
)


model = prepare_binary_model(
    model,
    bconfig,
    modules_mapping={
        EBConv2d: EBConv2d,
        nn.Linear: layers.Linear,
    },
)

# seed = 0
# state_path = f"/home/dev/data_main/LOGS/BNN/Group_Expert_React/ResNet_RSign/{seed}/model_best.pth.tar"
# print("Loading state:", state_path)
# state = torch.load(state_path, "cpu")
# model.load_state_dict(state["state_dict"], strict=False)
# with torch.no_grad():
#     print('Expanding the weights...')
#     for module in model.modules():
#         if isinstance(module, EBConv2d):
#             if not hasattr(module, 'activation_pre_process'):
#                 print("Skipped expert expansion for:", module)
#             else:
#                 if not isinstance(
#                         module.activation_pre_process,
#                         nn.Identity):
#                     print(
#                         f'Init module with w shape = {module.weight.size()}')
#                     for i in range(1, 4):
#                         module.weight.data[i, ...].copy_(
#                             module.weight.data[0, ...])


TARGET = "label"

run_experiment(model, get_tiny_imagenet_wds, TARGET)
