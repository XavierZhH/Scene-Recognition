# -*- coding: utf-8 -*-
import os
import random
import shutil

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

train_dir = 'training/'
validation_dir = 'training/'
test_dir = 'testing/'


class BoWData(Dataset):
    """
    Wrap the bow features into data that pytorch can traverse
    """

    def __init__(self, bows, targets):
        assert bows.size(0) == targets.size(0)
        total_num = bows.size(0)
        data_list = []
        for i in range(total_num):
            img = bows[i, :].type(torch.float32)
            target = targets[i].type(torch.long)
            data_list.append((img, target))

        self.data_list = data_list

    def __getitem__(self, item):
        img, target = self.data_list[item]
        return img, target

    def __len__(self):
        return len(self.data_list)


class TestImgData(Dataset):
    """
    Read test images and get their file names to write text
    """

    def __init__(self, test_dir, transform=None):
        img_names = os.listdir(test_dir)
        img_list = []
        for img_name in img_names:
            img = os.path.join(test_dir, img_name)
            img_list.append((Image.open(img).convert('RGB'), img_name))

        self.img_list = sorted(img_list, key=lambda v: v[1])
        self.transform = transform

    def __getitem__(self, item):
        img, name = self.img_list[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, name

    def __len__(self):
        return len(self.img_list)


def generate_validate_data(validation_dir, train_dir, ratio=0.2):
    """
    Randomly generate validation data from the training data.
    This function can be run once.
    :param validation_dir: the path where validation data is stored
    :param train_dir: the path where training data is stored
    :param ratio: generation ratio
    :return: void
    """
    if not os.path.exists(validation_dir):
        labels = datasets.ImageFolder(root=train_dir).classes
        select_range = list(range(0, 100))
        for label in labels:
            src_dir = os.path.join(train_dir, label)
            dst_dir = os.path.join(validation_dir, label)

            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            selected = random.sample(select_range, int(ratio * len(select_range)))
            for s in selected:
                img = os.path.join(src_dir, str(s) + '.jpg')
                img_dst = os.path.join(dst_dir, str(s) + '.jpg')
                shutil.copyfile(img, img_dst)


def get_test_data(test_dir, transform=None):
    """
    Get native test images.
    :param test_dir: the path where test data is stored
    :param transform: transform function
    :return: test images information list
    """
    img_names = os.listdir(test_dir)
    test_images = []
    for img_name in img_names:
        img = os.path.join(test_dir, img_name)
        if transform is not None:
            test_images.append((transform(Image.open(img).convert('RGB')), img_name))
        else:
            test_images.append((Image.open(img).convert('RGB'), img_name))
    return sorted(test_images, key=lambda v: v[1])


def get_labels(dir, imageFolder=None):
    """
    Get the correspondence between category numbers and tag names.
    :param dir: the path where training data is stored
    :param imageFolder: pytorch data structure
    :return: the correspondence between category numbers and tag names
    """
    if imageFolder is None:
        labels = datasets.ImageFolder(root=dir).classes
    else:
        labels = imageFolder.classes
    class_num_label = {k: labels[k] for k in range(0, len(labels))}
    return class_num_label


def zero_mean_and_unit_length(tensor):
    """
    zero mean and unit length conversion
    :param tensor: target tensor
    :return: transformed tensor
    """
    zero_mean = tensor - torch.mean(tensor).item()
    result = zero_mean / torch.norm(zero_mean, 2).item()
    return result
