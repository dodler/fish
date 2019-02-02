import os.path as osp
import random
from tqdm import *
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

bb = pd.read_csv('bounding_boxes.csv')


def read_img(img_path, crop=True):
    img_name = img_path.split('/')[-1]
    box = bb[bb.Image == img_name]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if crop:
        img = img[box.y0.values[0]:box.y1.values[0], box.x0.values[0]:box.x1.values[0], :]

    return img


def to_tensor(img):
    img = img.astype(np.float32) / 255.0 - 0.5
    img = img[:, :, (2, 1, 0)]  # BGR -> RGB
    return torch.from_numpy(img).permute(2, 0, 1)


class SeameseFishDs:
    def __init__(self, df, path, aug, positive_pair_num=5, negative_pair_num=15):
        self.positive_pair_num = positive_pair_num
        self.negative_pair_num = negative_pair_num
        self.df = df
        print(df.head())
        self.path = path
        self.aug = aug

        self.pairs = []
        self.make_pairs()

    def make_pairs(self):

        pair2img = {}
        for i in tqdm(range(len(self.df))):
            idx = self.df.Id[i]
            if idx not in pair2img.keys():
                pair2img[idx] = []

            pair2img[idx].append(self.df.Image[i])

        pair_img_keys = list(pair2img.keys())
        labels_lens = []
        for k in pair_img_keys:
            labels_lens.append(len(pair2img[k]))

        labels = []

        for idx in tqdm(pair2img.keys()):
            for i in range(self.positive_pair_num):
                pair_idx = idx
                label = 1.0
                labels.append(label)
                img = osp.join(self.path, random.choice(pair2img[idx]))
                pair_img = osp.join(self.path, random.choice(pair2img[pair_idx]))
                self.pairs.append((img, pair_img, label))

            for i in range(self.negative_pair_num):
                pair_idx = random.choice(pair_img_keys)
                if pair_idx == idx:
                    continue
                labels.append(0)
                img = osp.join(self.path, random.choice(pair2img[idx]))
                pair_img = osp.join(self.path, random.choice(pair2img[pair_idx]))

                self.pairs.append((img, pair_img, 0))

        print(np.histogram(labels, bins=3))
        print('pairs done,', self.pairs[0:5], ',')

    def __len__(self):
        return len(self.pairs)

    def choise_same_pair(self, idx):
        new_item = random.randrange(len(self.df.Id))
        while idx != self.df.Id[new_item]:
            new_item = random.randrange(len(self.df.Id))

        return new_item

    def choose_another_pair(self, idx):
        new_item = random.randrange(len(self.df.Id))
        while idx == self.df.Id[new_item]:
            new_item = random.randrange(len(self.df.Id))

        return new_item

    def __getitem__(self, item):

        img, img_pair, label = self.pairs[item]

        img = read_img(img)
        img_pair = read_img(img_pair)

        if self.aug is not None:
            img = self.aug(image=img)['image']
            img_pair = self.aug(image=img_pair)['image']

        img = to_tensor(img)
        img_pair = to_tensor(img_pair)

        label = torch.ones(1) * label

        return (img, img_pair), label


class FishDs:
    def __init__(self, df, path, aug):
        self.df = df
        self.path = path
        self.aug = aug

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        img_path = osp.join(self.path, self.df.Image[item])
        img = read_img(img_path)

        if self.aug is not None:
            img = self.aug(image=img)['image']

        img = to_tensor(img)

        label = torch.Tensor([self.df.Id[item].astype(np.long)]).long()

        return img, label


def get_seamese_ds(train_aug=None, valid_aug=None, path='/home/lyan/Documents/fish',
                   encode_labels=False,
                   drop_new_whale=True):
    train = pd.read_csv(osp.join(path, 'train.csv'))

    if drop_new_whale:
        train = train[train.Id != 'new_whale']

    encoder = None
    if encode_labels:
        encoder = LabelEncoder()
        train.Id = encoder.fit_transform(train.Id)

    train, valid = train_test_split(train)
    train.reset_index(inplace=True)
    valid.reset_index(inplace=True)

    train_ds = SeameseFishDs(train, osp.join(path, 'train'), train_aug)
    valid_ds = SeameseFishDs(valid, osp.join(path, 'train'), valid_aug)

    if encode_labels:
        return train_ds, valid_ds, encoder
    else:
        return train_ds, valid_ds


def get_classif_ds(train_aug=None, valid_aug=None,
                   path='/home/lyan/Documents/fish',
                   encode_labels=True,
                   drop_new_whale=True):
    train = pd.read_csv(osp.join(path, 'train.csv'))

    if drop_new_whale:
        print('shape before dropping', train.Id.shape)
        train = train[train.Id != 'new_whale']
        print('shape after dropping', train.Id.shape)

    if encode_labels:
        encoder = LabelEncoder()
        train.Id = encoder.fit_transform(train.Id)

    train, valid = train_test_split(train)
    train.reset_index(inplace=True)
    valid.reset_index(inplace=True)

    train_ds = FishDs(train, osp.join(path, 'train'), train_aug)
    valid_ds = FishDs(valid, osp.join(path, 'train'), valid_aug)

    if encode_labels:
        return train_ds, valid_ds, encoder
    else:
        return train_ds, valid_ds
