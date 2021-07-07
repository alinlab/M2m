import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms, utils, datasets
import random
import numpy as np

from torch import cuda
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler

image_size = 32
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
}

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


def default_loader(path):
    return Image.open(path).convert('RGB')


def get_celeb_loader(batch_size, mode=False, smote=False, num_workers=16):
    txt_train = './CelebA/celebA_train_orig.txt'
    txt_val = './CelebA/celebA_val_orig.txt'
    txt_test = './CelebA/celebA_test_orig.txt'

    data_root = '/home/temp/data/CelebA/'

    set_train = LT_Dataset(data_root, txt_train, data_transforms['train'])
    set_val = LT_Dataset(data_root, txt_val, data_transforms['val'])
    set_test = LT_Dataset(data_root, txt_test, data_transforms['test'])

    train_loader = DataLoader(set_train, batch_size, shuffle=True, num_workers=num_workers,pin_memory=cuda.is_available())
    val_loader = DataLoader(set_val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=cuda.is_available())
    test_loader = DataLoader(set_test, batch_size, shuffle=False, num_workers=num_workers, pin_memory=cuda.is_available())

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    for mode in ["train", "val", "test"]:
        loader = get_celeb_loader(128, mode, 4)
