from local_models.mnist import decomposed_cnn, two_layer_cnn
from local_datasets.dataset_makers import get_mnist, fashionMNIST
from bnn import BConfig, prepare_binary_model, Identity
from run_experiment import run_experiment

from bnn.ops import (
    BasicBinarizer,
    LearnableScale,
    LearnableBiasScale,
)

model = two_layer_cnn()

bconfig = BConfig(
    activation_pre_process=BasicBinarizer,
    activation_post_process=Identity,
    weight_pre_process=BasicBinarizer,
)


model = prepare_binary_model(
    model, bconfig, custom_config_layers_name={"fc": BConfig()},
)
print(model)

print('ALPHA INIT',model.alpha_one)
print('ALPHA INIT',model.alpha_two)

target = "label"
run_experiment(model, get_mnist, target)

print('ALPHA TRAINED',model.alpha_one)
print('ALPHA TRAINED',model.alpha_two)


