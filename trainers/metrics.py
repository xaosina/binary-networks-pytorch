import torch
from functools import partial
import inspect
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)

# a wrapper function
def add_target(f, TARGET):
    return partial(f, TARGET=TARGET)


# Define main functions


def rmse(outputs, batch, TARGET):
    outputs = outputs["preds"].cpu()
    return (
        mean_squared_error(batch[TARGET].detach().cpu(), outputs.detach().cpu())
        ** 0.5
    )


def mse_loss(outputs, batch, TARGET):
    return torch.nn.functional.mse_loss(
        outputs["preds"].view_as(batch[TARGET]), batch[TARGET]
    )


def cross_entropy(outputs, batch, TARGET):
    entropy = torch.nn.CrossEntropyLoss()
    return entropy(outputs["preds"], batch[TARGET])

def cross_entropy_segmentation(outputs, batch, TARGET):
    entropy = torch.nn.CrossEntropyLoss(ignore_index=255)
    return entropy(outputs["preds"], batch[TARGET])


def f1_macro(outputs, batch, TARGET):
    _, preds = torch.max(outputs["preds"], 1)
    preds = preds.cpu()
    return f1_score(batch[TARGET].cpu(), preds, average="macro")


def f1_weighted(outputs, batch, TARGET):
    _, preds = torch.max(outputs["preds"], 1)
    preds = preds.cpu()
    return f1_score(batch[TARGET].cpu(), preds, average="weighted")


def accuracy(outputs, batch, TARGET):
    _, preds = torch.max(outputs["preds"], 1)
    preds = preds.cpu()
    return accuracy_score(batch[TARGET].cpu(), preds)

def r2(outputs, batch, TARGET):
    return(r2_score(batch[TARGET].cpu(), outputs.cpu()))


class AucScore:
    def __init__(self, TARGET):
        self.TARGET = TARGET
        self.history_preds = []
        self.history_yhat = []

    def reset(self):
        self.history_preds = []
        self.history_yhat = []

    def update(self, outputs, batch):
        preds = outputs["preds"][:, 1]
        preds = preds.detach().cpu().tolist()
        yhat = batch[self.TARGET].cpu().tolist()
        self.history_preds += preds
        self.history_yhat += yhat
        try:
            score = roc_auc_score(self.history_yhat, self.history_preds)
        except Exception as e:
            print(e)
            score = 0.5

        return score

class IoU_pascal:
    def __init__(self, TARGET):
        self.TARGET = TARGET
        self.class_names = [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
            "motorbike", "person", "potted-plant", "sheep", "sofa", "train", "tv/monitor",
        ]
        self.num_classes = len(self.class_names)

    def reset(self):
        self.inter_sum = 0
        self.union_sum = 0

    def get_str(self):
        res = ""
        iou = self.inter_sum / (self.union_sum + 1e-10)
        for i, name in enumerate(self.class_names):
            res += f"{name}: {iou[i].round(3)}|"
        return res

    def update(self, outputs, batch):
        _, preds = torch.max(outputs["preds"], 1)
        inter, union = self.inter_and_union(preds, batch[self.TARGET])
        self.inter_sum += inter
        self.union_sum += union
        return (self.inter_sum / (self.union_sum + 1e-10)).mean()

    def inter_and_union(self, pred, mask):
        num_class = self.num_classes
        pred = pred.cpu().numpy().astype(np.uint8)
        mask = mask.cpu().numpy().astype(np.uint8)

        # 255 -> 0
        pred += 1
        mask += 1
        pred = pred * (mask > 0)

        inter = pred * (pred == mask)
        (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
        (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
        (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
        area_union = area_pred + area_mask - area_inter
        num_class = self.num_classes

        return (area_inter, area_union)

LOSSES = {
    "CrossEntropyLoss": cross_entropy, 
    "CrossEntropyLossSegment": cross_entropy_segmentation, 
    "MSELoss": mse_loss
}


METRICS = {
    "f1_macro": (f1_macro, "average"),
    "f1_weighted": (f1_weighted, "average"),
    "accuracy": (accuracy, "average"),
    "auc": (AucScore, "full"),
    "iou_pascal": (IoU_pascal, "full"),
    "rmse": (rmse, "average"),
    "r2": (r2, "average"),
}


def get_metrics_and_loss(loss_name, metric_names, target_name):
    loss_fn = add_target(LOSSES[loss_name], target_name)

    metrics = []

    for m_name in metric_names:
        f, t = METRICS[m_name]
        if inspect.isclass(f): #"auc" in m_name:
            metrics.append((m_name, add_target(f, target_name)(), t))
        else:
            metrics.append((m_name, add_target(f, target_name), t))

    return loss_fn, metrics
