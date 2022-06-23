from models.BlockExpertNet import resnet18, BasicBlock
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

ignore_layers_name = [
    '$layer+[0-9]\.+[0-9]\.fc$']

model = prepare_binary_model(
    model,
    bconfig,
    custom_config_layers_name={
        "first_conv": BConfig(),
        # "fc": BConfig(),
    },
    ignore_layers_name=ignore_layers_name,
)

seed = 0
state_path = f"/home/dev/data_main/LOGS/BNN/Group_Expert_React/ResNet_RSign/{seed}/model_best.pth.tar"
print("Loading state:", state_path)
state = torch.load(state_path, "cpu")
model.load_state_dict(state["state_dict"], strict=False)
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

run_experiment(model, get_tiny_imagenet_wds, TARGET)
