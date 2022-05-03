from re import L
import time
import torch

from trainers.optimizers import OPT, SCH
from tqdm.autonotebook import tqdm
from trainers.trainer_utils import AverageMeter, Timer
from omegaconf import OmegaConf as omg
from omegaconf.dictconfig import DictConfig 


class Trainer:
    def __init__(
        self,
        criterion,
        metrics,
        optimizers={"main": ("SGD", {"lr": 1e-3})},  # a list of tuples
        phases=["train"],
        num_epochs=3,
        device="cpu",
        logger=None,
        log_arch=False,
        dataset_opt_link=None,
        log_training=False,
        trainer_name="#",
    ):

        # can be a list of optimizers / probably need a dictionary ??
        self.dataloaders = None
        self.criterion = criterion
        self.phases = phases
        self.init_metrics = metrics
        self.metrics = self._init_metrics_and_loss(metrics)  # DICT of metrics
        # TODO change to be a dictionary
        if type(optimizers) == DictConfig:
            optimizers = omg.to_container(optimizers)
        self.initial_optimizers = optimizers
        self.num_epochs = num_epochs
        self.device = device
        self.logger = logger
        self.logging = False if logger is None else True
        self.log_arch = log_arch
        self.log_training = log_training
        self.trainer_name = trainer_name
        self.dataset_opt_link = dataset_opt_link

    def set_model(self, model, param_groups="default"):
        # TODO: update optimizer prameters dict
        self.model = model
        self.model.to(self.device)
        self._log(self.model)
        if param_groups == "default":
            param_groups = {"main": self.model.parameters()}
        self._log("Use GPU: {} for training".format(self.device))
        self._init_optimizers(param_groups)
        self.metrics = self._init_metrics_and_loss(self.init_metrics)

    # TODO it does not work with LR scheduler
    # See method usage in Random searcher
    def _init_optimizers(self, param_groups: dict()):
        self.optimizers = dict()
        self.schedulers = dict()
        if not isinstance(param_groups, dict):
            param_groups = self._dictify(param_groups, initial=["main"])

        # TODO check that keys in opt_groups match keys in self.initial_optimizers
        for opt_group in param_groups:
            opt_name, opt_params = self.initial_optimizers[opt_group]["optimizer"]
            sch_name, sch_params = self.initial_optimizers[opt_group]["scheduler"]
            if opt_params is None:
                opt_params = dict()
            if sch_params is None:
                sch_params = dict()
            opt_params["params_dict"] = param_groups[opt_group]
            if sch_name == "cosine": 
                print(sch_params)
                sch_params["epochs"] = self.num_epochs
            self.optimizers[opt_group] = OPT[opt_name](**opt_params)
            sch_params["optimizer"] = self.optimizers[opt_group]
            self.schedulers[opt_group] = SCH[sch_name](**sch_params)

    # TODO Loaders should match training phases, add check

    def set_dataloaders(self, dataloaders):
        self.dataloaders = dataloaders

    def _init_metrics_and_loss(self, metrics):
        metrics_dict = dict()
        for phase in self.phases:
            metrics_dict[phase] = {
                n: AverageMeter(m, n, t) for (n, m, t) in metrics
            }
            # if phase == "train":
            metrics_dict[phase]["loss"] = AverageMeter(
                self.criterion, "loss", "average"
            )
        return metrics_dict

    def _update_metrics(self, outputs, batch, phase):
        for name in self.metrics[phase]:
            self.metrics[phase][name](outputs, batch)

    def _opt_zero_grad(self, phase):
        self._iter_and_call_optim("zero_grad", phase)

    def _opt_step(self, phase):
        self._iter_and_call_optim("step", phase)

    def _iter_and_call_optim(self, method_name, phase):
        if phase != "validation":
            if isinstance(self.optimizers, dict):
                if self.dataset_opt_link is None:
                    for name in self.optimizers:
                        method = getattr(self.optimizers[name], method_name)
                        method()
                else:
                    name = self.dataset_opt_link[phase]
                    method = getattr(self.optimizers[name], method_name)
                    method()

            else:
                method = getattr(self.optimizers, method_name)
                method()

    def _scheduler_step(self, phase):
        if phase != "validation":
            if isinstance(self.schedulers, dict):
                if self.dataset_opt_link is None:
                    for name in self.schedulers:
                        method = self.schedulers[name].step
                        method()
                        self._log(f"New lr for group '{name}': {self.schedulers[name].get_last_lr()}")
                else:
                    name = self.dataset_opt_link[phase]
                    method = self.schedulers[name].step
                    method()
                    self._log(f"New lr for group '{name}': {self.schedulers[name].get_last_lr()}")
            else:
                method = self.schedulers.step
                method()
                self._log(f"New lr: {self.schedulers.get_last_lr()}")


    def _log(self, message):
        if self.logging:
            self.logger.log(message=message)
        else:
            print(message)

    def _dictify(self, object, initial=[]):

        # TODO check if object is iterable

        if isinstance(object, dict):
            return object
        else:
            # Assume inputs and targets go first
            labels = initial
            if len(object) - len(initial) > 0:
                # Numerate other by keys if keys were not provided
                labels += list(range(len(object) - len(labels)))
            return {k: v for (k, v) in zip(labels, object)}

    def _data_to_device(self, data, device):

        # recursivly set to device
        def to_device(obj):
            for key in obj:
                if isinstance(obj[key], dict):
                    to_device(obj[key])
                else:
                    obj[key] = obj[key].to(device)

        to_device(data)

        return data

    def _process_batch(self, batch):
        batch = self._dictify(batch, initial=["model_input", "target"])
        batch = self._data_to_device(batch, self.device)
        return batch

    # def _iterate_dataset():
    def _log_arch(self, epoch):
        if self.logging:
            if self.log_arch:
                if hasattr(self.model, "get_arch"):
                    for key in self.model.get_arch():
                        print(
                            f"Current arch weights {key}",
                            self.model.get_arch()[key],
                        )
                        values = {
                            str(k): self.model.get_arch()[key][0][k]
                            for k in range(
                                0, len(self.model.get_arch()[key][0])
                            )
                        }
                        self.logger.log_metrics("arch_" + key, values, epoch)

    def _log_metrics(self, epoch):
        if self.logging:
            metrics_val = self.get_last_epoch_metrics(phase="validation")
            metrics_train = self.get_last_epoch_metrics(phase="train")

            self.logger.log_metrics(
                f"{self.trainer_name }_val", metrics_val, epoch
            )
            self.logger.log_metrics(
                f"{self.trainer_name }_train", metrics_train, epoch
            )

    def _iterate_one_epoch(self, phase, epoch):
        # Each epoch has a training and validation phase

        if phase == "validation":
            self.model.eval()
        else:
            self.model.train()

        # Iterate over data.

        n_batches = len(self.dataloaders[phase])
        end = time.time()
        data_time = Timer("Data")
        batch_time = Timer("Time")
        with tqdm(self.dataloaders[phase], total=n_batches) as t:
            for batch in t:
                data_time.update(time.time() - end)
                batch = self._process_batch(batch)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase.startswith("train")):
                    outputs = self.model(batch["model_input"])
                    # TODO move metrics to cpu and keep loss on gpu

                    self._update_metrics(outputs, batch, phase)

                    # backward + optimize only if in training phase
                    if phase.startswith("train"):
                        self.metrics[phase]["loss"].last_value.backward()
                        self._opt_step(phase)
                        self._opt_zero_grad(phase)
                batch_time.update(time.time() - end)
                for_print = {
                    batch_time.name: str(batch_time), 
                    data_time.name: str(data_time)
                }
                for_print.update({
                    k: "{:.3f}".format(self.metrics[phase][k].avg) for k in self.metrics[phase]
                })
                for_print = "|".join([" ".join([k, for_print[k]]) for k in for_print])
                t.set_description(for_print)
                end = time.time()
        self._log(for_print)
        for_board = {
            batch_time.name: batch_time.sum, 
            data_time.name: data_time.sum
        }
        self.logger.add_scalars(f"{phase} time", for_board, epoch)
        self._scheduler_step(phase)

        # use as a dummy input to compute a graph
        self.last_batch = batch

    def terminator_chek_if_nan_or_inf(self):
        """Get last loss values if inf or nan terminate"""
        flag = False
        for phase in self.phases:
            loss = self.get_history(phase=phase)["loss"]
            loss = torch.tensor(loss)
            if any(~torch.isfinite(loss)) or any(torch.isnan(loss)):
                self.logger.log("TERMINATING NA or INF in loss encountered")
                flag = False

        return flag

    def train(self):
        start_time = time.time()
        best_acc = 0
        for epoch in range(self.num_epochs):
            self._log_arch(epoch)
            self._log_metrics(epoch)
            for phase in self.phases:
                self.reset_metrics(phase)
                self._log(" Epoch {}/{}".format(epoch, self.num_epochs - 1))
                self._iterate_one_epoch(phase, epoch)
                metrics = self.get_last_epoch_metrics(phase)
                metric_string = " | ".join(
                    [f"{key}:{value:.3f}" for key, value in metrics.items()]
                )
                self._log(f"{phase.upper()} {metric_string}")
                if phase == "validation": 
                    state = {
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'metric': metrics["accuracy"] if "accuracy" in metrics else 0,
                    }
                    is_best = state["metric"] >= best_acc
                    best_acc = state["metric"] if state["metric"] >= best_acc else best_acc
                    self.logger.save_checkpoint(state, is_best)
                    
            if self.terminator_chek_if_nan_or_inf():
                break

        time_elapsed = time.time() - start_time
        self._log(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    def get_history(self, phase="validation"):
        history = dict()
        for k in self.metrics[phase]:
            history[k] = self.metrics[phase][k].history

        return history

    def get_last_epoch_metrics(self, phase="validation"):
        avg = dict()
        for k in self.metrics[phase]:
            avg[k] = self.metrics[phase][k].avg

        return avg

    def reset_metrics(self, phase="validation"):
        for k in self.metrics[phase]:
            self.metrics[phase][k]._reset()
