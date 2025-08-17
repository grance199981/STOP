import os

from torchvision import datasets, transforms
import torch
from models.functions.data_loaders import build_dvscifar
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST

def load_dataset(args, dataset='MNIST', batch_size=100, dataset_path='../../data', is_cuda=False):
    kwargs = {'num_workers': 4, 'pin_memory': True} if is_cuda else {}
    if dataset == 'MNIST':
        num_classes = 10
        dataset_train = datasets.MNIST(os.path.join(dataset_path, 'MNIST'), train=True, download=True,
                                       transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(dataset_path, 'MNIST'), train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

    elif dataset == 'CIFAR10':
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
        ])
        dataset_train = datasets.CIFAR10(os.path.join(dataset_path, 'CIFAR10'), train=True, download=False,
                                         transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(dataset_path, 'CIFAR10'), train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                             ])),
            batch_size=batch_size, shuffle=False, **kwargs)
    elif dataset == 'CIFAR100':
        num_classes = 100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        dataset_train = datasets.CIFAR100(os.path.join(dataset_path, 'CIFAR100'), train=True, download=False,
                                          transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(dataset_path, 'CIFAR100'), train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                              ])),
            batch_size=batch_size, shuffle=False, **kwargs)
    elif dataset == 'ImageNet':
        num_classes = 1000
        traindir = os.path.join(dataset_path, 'train')
        valdir = os.path.join(dataset_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False, **kwargs)
    elif dataset == 'NMNIST':
        num_classes = 10


        train_dataset = NMNIST(root=os.path.join(dataset_path, 'NMNIST'), train=True,data_type='frame',
                                                   frames_number=args.time_window, split_by='number')

        val_dataset = NMNIST(root=os.path.join(dataset_path, 'NMNIST'), train=False,data_type='frame',
                                                   frames_number=args.time_window, split_by='number')


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=4, pin_memory=True)

    elif dataset == 'DVSCIFAR10':
        num_classes = 10
        # Data loading code
        train_dataset, val_dataset = build_dvscifar(path=os.path.join(dataset_path, 'DVSCIFAR10'), transform=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   **kwargs)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                  **kwargs)
    elif dataset == 'DVSGesture':

        num_classes = 11


        train_dataset = DVS128Gesture(root=os.path.join(dataset_path, 'DVSGesture'), train=True,data_type='frame',
                                                   frames_number=args.time_window, split_by='number')

        val_dataset = DVS128Gesture(root=os.path.join(dataset_path, 'DVSGesture'), train=False,data_type='frame',
                                                   frames_number=args.time_window, split_by='number')


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   **kwargs)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                 **kwargs)

    else:
        raise Exception('No valid dataset is specified.')
    return train_loader, test_loader, num_classes
