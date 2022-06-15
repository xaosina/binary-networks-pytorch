from models.expert_react_group_net import resnet18, BasicBlock
import torch

from datasets.datasets import get_tiny_image_net
from bnn import BConfig, prepare_binary_model, Identity
from run_experiment import run_experiment

from bnn.ops import (
    BasicInputBinarizer,
    XNORWeightBinarizer,
    BasicScaleBinarizer,
    InputBiasBinarizer
    
)

model = resnet18(num_classes=200, single_path=True, use_only_first=True)

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


state = torch.load("/home/dev/data_main/LOGS/BNN/group_net_updates/RSign_only_first_expert/model_best.pth.tar", "cpu")
model.load_state_dict(state["state_dict"])
with torch.no_grad():
    for module in model.named_modules():
        if isinstance(module[1], BasicBlock):
            m = module[1]
            print(module[0])
            for mlist in [m.conv1, m.bn1, m.relu1, m.conv2, m.bn2, m.relu2, m.scales]:
                for i in range(1, m.num_bases):
                    if isinstance(mlist, torch.nn.ParameterList):
                        mlist[i].copy_(mlist[0])
                    else:
                        mlist[i].load_state_dict(mlist[0].state_dict())


TARGET = "label"

run_experiment(model, get_tiny_image_net, TARGET)
