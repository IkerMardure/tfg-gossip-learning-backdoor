import numpy as np
import pandas as pd
import torch
import sys
import os
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms, utils
import os
from itertools import chain
import torchvision.transforms
import torchvision.datasets as torch_datasets
from utils.paths import resolve_data_path
#import ssl


def _default_num_workers() -> int:
    # Ray workers on Windows are much more stable with single-process data loading.
    return 0 if os.name == "nt" else 4


def _make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int | None = None):
    workers = _default_num_workers() if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _group_dataset_by_label(dataset, num_classes: int):
    buckets = [[] for _ in range(num_classes)]
    for sample in dataset:
        label = int(sample[1])
        if 0 <= label < num_classes:
            buckets[label].append(sample)
    return buckets


def get_cifar10(data_path: str | None = None):
    """Download CIFAR‑10 and apply a simple transform."""
    #ssl._create_default_https_context = ssl._create_unverified_context
    torch_datasets.CIFAR10.url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    transform_train = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
            transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
            transforms.RandomRotation(10),  # Rotates the image to a specified angel
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    resolved = resolve_data_path(data_path)
    trainset = torch_datasets.CIFAR10(
        str(resolved), train=True, download=True, transform=transform_train
    )
    testset = torch_datasets.CIFAR10(
        str(resolved), train=False, download=True, transform=transform_test
    )

    return trainset, testset


# ---------- MNIST support ----------

def get_mnist(data_path: str | None = None):
    """Download MNIST and apply minimal transform using torchvision."""
    transform_train = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    resolved = resolve_data_path(data_path)
    trainset = torch_datasets.MNIST(
        str(resolved), train=True, download=True, transform=transform_train
    )
    testset = torch_datasets.MNIST(
        str(resolved), train=False, download=True, transform=transform_test
    )
    return trainset, testset


def prepare_dataset_mnist_iid(
    num_clients: int,
    num_classes: int,
    clients_with_no_data: list[int],
    batch_size: int,
    seed: int,
    data_path: str | None = None,
    val_ratio: float = 0.1,
):
    """Load MNIST and split IID across clients."""
    trainset, testset = get_mnist(data_path)
    return _split_iid(
        trainset,
        testset,
        num_clients,
        num_classes,
        clients_with_no_data,
        batch_size,
        seed,
        val_ratio,
    )



# helper used by IID/NIID functions to split a given dataset

def _split_iid(
    trainset,
    testset,
    num_clients: int,
    num_classes: int,
    clients_with_no_data: list[int],
    batch_size: int,
    seed: int,
    val_ratio: float = 0.1,
):
    clients_with_data = []
    for i in range(num_clients):
        if i not in clients_with_no_data:
            clients_with_data.append(i)

    ordered_trainset = [item for class_data in _group_dataset_by_label(trainset, num_classes) for item in class_data]

    num_images = len(ordered_trainset) // len(clients_with_data)
    num_images_remainder = len(ordered_trainset) % len(clients_with_data)

    partition_len = [0] * num_clients

    # SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len[i] = num_images
        if num_images_remainder > 0:
            partition_len[i] += 1
            num_images_remainder -= 1

    ##########
    trainsets = random_split(
        ordered_trainset, partition_len, torch.Generator().manual_seed(seed)
    )
    trainloaders = []
    validationloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(seed)
        )
        if num_total > 0:
            trainloaders.append(_make_loader(for_train, batch_size=batch_size, shuffle=True))
            validationloaders.append(_make_loader(for_val, batch_size=batch_size, shuffle=False))
        else:
            trainloaders.append('')
            validationloaders.append('')

    ordered_testset = [item for class_data in _group_dataset_by_label(testset, num_classes) for item in class_data]

    testloader = _make_loader(ordered_testset, batch_size=batch_size, shuffle=True)
    return trainloaders, validationloaders, testloader


def prepare_dataset_iid(
    num_clients: int,
    num_classes: int,
    clients_with_no_data: list[int],
    batch_size: int,
    seed: int,
    val_ratio: float = 0.1,
):
    """Load CIFAR-10 (training and test set)."""
    trainset, testset = get_cifar10()
    return _split_iid(
        trainset,
        testset,
        num_clients,
        num_classes,
        clients_with_no_data,
        batch_size,
        seed,
        val_ratio,
    )

def prepare_dataset_niid(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, val_ratio: float = 0.1):
    """Load CIFAR-10 (training and test set). DIRICHLET"""
    trainset, testset = get_cifar10()

    clients_with_data = []
    for i in range(num_clients):
        if i not in clients_with_no_data:
            clients_with_data.append(i)

    ordered_trainset = [item for class_data in _group_dataset_by_label(trainset, num_classes) for item in class_data]

    # SPLIT DIRICHLET DISTRIBUTION
    alpha = 0.4
    np.random.seed(seed=seed)
    dirich = np.random.dirichlet([alpha]*len(clients_with_data))

    #num_images = len(ordered_trainset) // len(clients_with_data)
    #num_images_remainder = len(ordered_trainset) % len(clients_with_data)
    
    partition_len = [0] * num_clients
    total_instances = 0
    j = 0
    
    #SPLIT DS ACCORDINGLY
    for i in clients_with_data:
        partition_len[i] = int(len(ordered_trainset)*dirich[j])
        total_instances += partition_len[i]
        j+=1

    remainder = len(ordered_trainset) - total_instances
    partition_len[clients_with_data[0]] += remainder

    ##########
    trainsets = random_split(
        ordered_trainset, partition_len, torch.Generator().manual_seed(seed)
    )
    trainloaders = []
    validationloaders = []
    
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(seed)
        )
        if num_total > 0:
            trainloaders.append(_make_loader(for_train, batch_size=batch_size, shuffle=True))
            validationloaders.append(_make_loader(for_val, batch_size=batch_size, shuffle=False))
        else:
            trainloaders.append('')
            validationloaders.append('')

    ordered_testset = [item for class_data in _group_dataset_by_label(testset, num_classes) for item in class_data]


    testloader = _make_loader(ordered_testset, batch_size=batch_size, shuffle=True)
    return trainloaders, validationloaders, testloader, partition_len


def prepare_dataset_niid_class_partition(num_clients: int, num_classes: int, clients_with_no_data: list[int], batch_size: int, seed: int, val_ratio: float = 0.1):
    """Load CIFAR-10 (training and test set)."""
    trainset, testset = get_cifar10()

    #num_images = len(trainset) // num_clients
    clients_with_data = []
    #partition_len = [0] * num_clients
    #num_classes = 10

    for i in range(num_clients):
        if i not in clients_with_no_data:
            clients_with_data.append(i)

    ordered_trainset = _group_dataset_by_label(trainset, num_classes)

    # Smart division
    partition_num_per_agent = num_classes // len(clients_with_data)
    partition_remainder_per_agent = num_classes % len(clients_with_data)

    trainsets = []
    for i in range(num_clients):
        tmp_list = []
        if i in clients_with_data:
            for j in range(partition_num_per_agent):
                tmp_list.extend(ordered_trainset.pop())
            if partition_remainder_per_agent > 0:
                tmp_list.extend(ordered_trainset.pop())
                partition_remainder_per_agent=-1
        trainsets.append(tmp_list)

    
    trainloaders = []
    validationloaders = []
    
    # VALIDATION SET
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(seed)
        )
        if num_total > 0:
            trainloaders.append(_make_loader(for_train, batch_size=batch_size, shuffle=True))
            validationloaders.append(_make_loader(for_val, batch_size=batch_size, shuffle=True))
        else:
            trainloaders.append('')
            validationloaders.append('')

    ordered_testset = [item for class_data in _group_dataset_by_label(testset, num_classes) for item in class_data]


    testloader = _make_loader(ordered_testset, batch_size=batch_size, shuffle=True)
    return trainloaders, validationloaders, testloader, partition_num_per_agent



def prepare_dataset_cnl(batch_size: int, seed: int, val_ratio: float = 0.1):
    """Load CIFAR-10 (training and test set)."""
    trainset, testset = get_cifar10()
    num_total = len(trainset)
    num_val = int(val_ratio * num_total)
    num_train = num_total - num_val
    for_train, for_val = random_split(
        trainset, [num_train, num_val], torch.Generator().manual_seed(seed)
    )
    trainloaders = _make_loader(for_train, batch_size=batch_size, shuffle=True)
    validationloaders = _make_loader(for_val, batch_size=batch_size, shuffle=True)    
    testloader = _make_loader(testset, batch_size=batch_size, shuffle=True)
    return trainloaders, validationloaders, testloader