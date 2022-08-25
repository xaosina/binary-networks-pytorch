import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import datasets, transforms
import webdataset as wds
from .pascal import VOCSegmentation


class LengthSet(IterableDataset):
    def __init__(self, dataset, lenght=None):
        self.dataset = dataset
        self.size = len(dataset) if lenght is None else lenght

    def __len__(self):
        return self.size
    
    def __iter__(self):
        return ({"model_input": b["model_input"], "label": b["label"]} for b in iter(self.dataset))

class DictSet(Dataset):
    def __init__(self, dataset, lenght=None):

        self.dataset = dataset
        self.size = len(dataset) 

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(int(index))
        return {"model_input": image, "label": label}

    def __len__(self):
        return self.size


def get_mnist(data_path=None, batch_size=-1, workers=-1):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset1 = DictSet(
        datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
    )

    dataset2 = DictSet(
        datasets.MNIST("../data", train=False, transform=transform)
    )

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=32)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=32)
    return train_loader, test_loader

def get_base_imagenet(batch_size=32, workers=4):
    return get_tiny_image_net("/home/dev/data_main/imagenet", batch_size=batch_size, workers=workers)

def get_tiny_image_net(data_path='/home/dev/data_main/CORESETS/TinyImagenet/tiny-224', batch_size=32, workers=4):
    # /home/dev/data_main/imagenet
 # Data loading code
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    crop_size, short_size = 224, 256
    train_dataset = DictSet(datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])))

    val_dataset = DictSet(datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(short_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader

def get_wds_set(data_path='/home/dev/data_main/CORESETS/TinyImagenet_wds', batch_size=32, workers=4):
    lenghts = {
        "/home/dev/data_main/CORESETS/TinyImagenet_wds": [100000, 5000],
        "/home/dev/data_main/imagenet_shards": [1281167, 50000],
        "/home/dev/data_main/imagenet_wds": [1281167, 50000] 
    }
    urls = {
        "/home/dev/data_main/CORESETS/TinyImagenet_wds": (
            "/imagenet-train-{000000..000009}.tar", 
            "/imagenet-val-{000000..000000}.tar"
            ),
        "/home/dev/data_main/imagenet_shards": (
            "/imagenet-train-{000000..000146}.tar", 
            "/imagenet-val-{000000..000006}.tar"
        ),
        "/home/dev/data_main/imagenet_wds": (
            "/imagenet-train-{000000..000009}.tar", 
            "/imagenet-val-{000000..000000}.tar"
        ) 
    }

    train_len, val_len = lenghts[data_path]
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    url = data_path + urls[data_path][0]
    train_dataset = LengthSet(
        wds.WebDataset(url)
        .decode("pil")
        .rename(model_input="jpg", label="cls")
        .map_dict(model_input=transforms_train, label=lambda x: torch.tensor(x))
        .shuffle(1000),
        lenght=train_len
    )
    url = data_path + urls[data_path][1]
    val_dataset = LengthSet(
        wds.WebDataset(url)
        .decode("pil")
        .rename(model_input="jpg", label="cls")
        .map_dict(model_input=transforms_val, label=lambda x: torch.tensor(x)), 
        lenght=val_len
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader

def get_tiny_imagenet_wds(batch_size=32, workers=4):
    # "/home/dev/data_main/CORESETS/TinyImagenet_wds": [100000, 5000],
    # "/home/dev/data_main/imagenet_fast": [1281167, 50000] 
    train_loader, val_loader = get_wds_set("/home/dev/data_main/CORESETS/TinyImagenet_wds", batch_size, workers)
    return train_loader, val_loader

def get_imagenet_wds(batch_size=32, workers=4):
    # "/home/dev/data_main/CORESETS/TinyImagenet_wds": [100000, 5000],
    # "/home/dev/data_main/imagenet_fast": [1281167, 50000] 
    train_loader, val_loader = get_wds_set("/home/dev/data_main/imagenet_shards", batch_size, workers)
    return train_loader, val_loader

def get_imagenet_wds_new(batch_size=32, workers=4):
    # "/home/dev/data_main/CORESETS/TinyImagenet_wds": [100000, 5000],
    # "/home/dev/data_main/imagenet_fast": [1281167, 50000] 
    train_loader, val_loader = get_wds_set("/home/dev/data_main/imagenet_wds", batch_size, workers)
    return train_loader, val_loader

def get_pascal(batch_size=64, workers=10):
    train_dataset = VOCSegmentation(
        "/home/dev/data_main/PASCAL", train=True, train_transform=True
    )
    test_dataset = VOCSegmentation(
        "/home/dev/data_main/PASCAL", train=False
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    return trainloader, testloader