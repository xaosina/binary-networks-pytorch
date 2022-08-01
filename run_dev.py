from models.BlockExpertNet import resnet18, BasicBlock
from models.Ensembles import ConcatHead, CustomConcat
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

model = resnet18(num_classes=200, rprelu=False, sampling={"k":1, "t":0.1})

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

seed = 2
state_path = f"/home/dev/data_main/LOGS/BNN/Group_Expert_React/BlockExpert_RSign_Corrected/{seed}/model_best.pth.tar"
print("Loading state:", state_path)
state = torch.load(state_path, "cpu")
model.load_state_dict(state["state_dict"], strict=False)

model = CustomConcat(model)


TARGET = "label"

run_experiment(model, get_tiny_imagenet_wds, TARGET)
