import os
import copy
import argparse
from omegaconf import OmegaConf as omg
import random
import warnings
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from trainers.trainer import Trainer
from nash_logging.common import LoggerUnited
from trainers.metrics import get_metrics_and_loss



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--exp_name', metavar='exp_name', default='test',
                    help='experiment name')

parser.add_argument('--gpu', metavar='gpu', default=None,
                    help='gpu')

parser.add_argument("-u", '--user', type=str, choices=["dima", "egor"], required=True)

parser.add_argument('--seed', metavar='seed', default=None,
                    help='seed')

args = vars(parser.parse_args())


def run_experiment(model, get_loaders, target):
    """
    The run_experiment function is the main function that runs your experiment.
    It takes in a model, a get_loaders function, and a target (which is either &quot;train&quot; or &quot;test&quot;).
    The model is trained on the training set using train_loader and evaluated on the test set using test_loader.
    The metrics are logged to both Tensorboard and Wandb. The best performing weights according to validation accuracy are saved as checkpoint files.
    
    :param model: Define the model that is going to be trained
    :param get_loaders: Get the dataloaders for training and testing
    :param target: Define the target of the experiment
    :return: A dictionary with the following keys:
    :doc-author: Trelent
    """
   
    env_config = omg.load(f"./configs/env_{args['user']}.yaml")
    
    if args['gpu'] is not None:
        env_config.HARDWARE.GPU = int(args['gpu'])

    if args["seed"] is not None:
        env_config.HARDWARE.SEED = int(args["seed"])

    new_conf = copy.deepcopy(env_config)
    new_conf.EXPERIMENT.DIR = os.path.join(
        env_config.EXPERIMENT.DIR, args['exp_name']
    )

    if new_conf.HARDWARE.SEED is not None:
        random.seed(new_conf.HARDWARE.SEED)
        np.random.seed(new_conf.HARDWARE.SEED)
        torch.manual_seed(new_conf.HARDWARE.SEED)
        cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
                        # 'This will turn on the CUDNN deterministic setting, '
                        # 'which can slow down your training considerably! '
                        # 'You may see unexpected behavior when restarting '
                        # 'from checkpoints.')
                      
    logger = LoggerUnited(new_conf, online_logger="tensorboard")
    logger.log(new_conf)
    logger.log('Num parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    train_loader, test_loader = get_loaders(workers=new_conf.HARDWARE.WORKERS,
                                            batch_size=int(new_conf.TRAINING['BATCH_SIZE']))

    dataloaders = {"train": train_loader, "validation": test_loader}

    TARGET = target
    criterion, metrics = get_metrics_and_loss(
        "CrossEntropyLoss", ["accuracy"], TARGET
    )

    trainer = Trainer(
        criterion,
        metrics,
        optimizers=new_conf.OPTIMIZER,
        phases=["train", "validation"],
        num_epochs=int(new_conf.TRAINING['EPOCHS']),
        device=env_config.HARDWARE.GPU,
        logger=logger,
        log_training=True,
    )

    trainer.set_model(model, {"main": filter(lambda p: p.requires_grad, model.parameters())})
    trainer.set_dataloaders(dataloaders)
    trainer.train()
    history = trainer.get_history()

    for k in history:
        for i, value in enumerate(history[k]):
            logger.log_metrics("Best val scores", {k: value}, i)
        print(k, max(history[k]))

    logger.shutdown_logging()




def validate(model, get_loaders, target):
    """
    Args:
        model (_type_): _description_
        get_loaders (_type_): _description_
        target (_type_): _description_
    """
   
    env_config = omg.load(f"./configs/env_{args['user']}.yaml")
    
    if args['gpu'] is not None:
        env_config.HARDWARE.GPU = int(args['gpu'])

    new_conf = copy.deepcopy(env_config)
    new_conf.EXPERIMENT.DIR = os.path.join(
        env_config.EXPERIMENT.DIR, args['exp_name']
    )

    if new_conf.HARDWARE.SEED is not None:
        random.seed(new_conf.HARDWARE.SEED)
        np.random.seed(new_conf.HARDWARE.SEED)
        torch.manual_seed(new_conf.HARDWARE.SEED)
        cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
                        # 'This will turn on the CUDNN deterministic setting, '
                        # 'which can slow down your training considerably! '
                        # 'You may see unexpected behavior when restarting '
                        # 'from checkpoints.')
                      
    logger = LoggerUnited(new_conf, online_logger="tensorboard")
    logger.log(new_conf)
    
    train_loader, test_loader = get_loaders(workers=new_conf.HARDWARE.WORKERS,
                                            batch_size=int(new_conf.TRAINING['BATCH_SIZE']))

    dataloaders = {"validation": test_loader}

    TARGET = target
    criterion, metrics = get_metrics_and_loss(
        "CrossEntropyLoss", ["accuracy"], TARGET
    )

    trainer = Trainer(
        criterion,
        metrics,
        optimizers=new_conf.OPTIMIZER,
        phases=["validation"],
        num_epochs=1,
        device=env_config.HARDWARE.GPU,
        logger=logger,
        log_training=True,
        trainer_name='validation'
    )

    model.to(env_config.HARDWARE.GPU)
    trainer.model = model
    trainer.set_dataloaders(dataloaders)
    trainer._iterate_one_epoch('validation')
    history = trainer.get_history()

    for k in history:
        for i, value in enumerate(history[k]):
            logger.log_metrics("Best val scores", {k: value}, i)
        print(k, max(history[k]))

    logger.shutdown_logging()