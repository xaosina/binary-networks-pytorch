import torch.utils.data as data
import os
from PIL import Image
import torch
import random
import math
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np


class VOCSegmentation(data.Dataset):
    # 21 class
    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted-plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    def __init__(
        self,
        root="/home/dev/data_main/PASCAL",
        train=True,
        transform=None,
        train_transform=False,
        target_transform=None,
        download=False,
        crop_size=513,
        resize=1
    ):
        self.root = root
        _voc_root = os.path.join(self.root, "VOC2012")
        _list_dir = os.path.join(_voc_root, "list")
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.crop_size = int(crop_size * resize) if crop_size is not None else None
        self.scale = (0.5, 2.0)
        self.train_transform = train_transform
        self.resize = resize

        if download:
            self.download()

        if self.train:
            _list_f = os.path.join(_list_dir, "train_aug.txt")
        else:
            _list_f = os.path.join(_list_dir, "val.txt")
        self.images = []
        self.masks = []
        with open(_list_f, "r") as lines:
            for line in lines:
                _image = _voc_root + line.split()[0]
                _mask = _voc_root + line.split()[1]
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.masks[index])

        if self.resize != 1:
            w, h = _img.size
            new_size = (int(round(w * self.resize)), int(round(h * self.resize)))
            _img = _img.resize(new_size, Image.ANTIALIAS)
            _target = _target.resize(new_size, Image.NEAREST)
        _img, _target = preprocess(
            _img,
            _target,
            flip=True if (self.train and self.train_transform) else False,
            scale=self.scale if (self.train and self.train_transform) else None,
            crop=(self.crop_size, self.crop_size)
            if self.crop_size is not None
            else None,
        )

        if self.transform is not None:
            _img = self.transform(_img)

        if self.target_transform is not None:
            _target = self.target_transform(_target)

        
        return {"model_input": _img, "label": _target}

    def __len__(self):
        return len(self.images)

    def download(self):
        raise NotImplementedError("Automatic download not yet implemented.")

def preprocess(image, mask, flip=False, scale=None, crop=None):
    if flip:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if scale:
        w, h = image.size
        rand_log_scale = math.log(scale[0], 2) + random.random() * (
            math.log(scale[1], 2) - math.log(scale[0], 2)
        )
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        image = image.resize(new_size, Image.ANTIALIAS)
        mask = mask.resize(new_size, Image.NEAREST)

    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = data_transforms(image)
    mask = torch.LongTensor(np.array(mask).astype(np.int64))

    if crop:
        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, crop[0] - h)
        pad_lr = max(0, crop[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        h, w = image.shape[1], image.shape[2]
        i = random.randint(0, h - crop[0])
        j = random.randint(0, w - crop[1])
        image = image[:, i : i + crop[0], j : j + crop[1]]
        mask = mask[i : i + crop[0], j : j + crop[1]]

    return image, mask
