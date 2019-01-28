import os.path as osp
import random
from tqdm import *
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def to_tensor(img):
    img = img.astype(np.float32) / 255.0 - 0.5
    img = img[:, :, (2, 1, 0)]  # BGR -> RGB
    return torch.from_numpy(img).permute(2, 0, 1)


class SeameseFishDs:
    def __init__(self, df, path, aug, min_num_pairs=5):
        self.min_num_pairs = min_num_pairs
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
            pair_idx = idx
            label = float(pair_idx == idx)

            labels.append(label)

            img = osp.join(self.path, random.choice(pair2img[idx]))
            pair_img = osp.join(self.path, random.choice(pair2img[pair_idx]))

            self.pairs.append((img, pair_img, label))
            for i in range(self.min_num_pairs):
                pair_idx = random.choice(pair_img_keys)
                label = 1-float(pair_idx == idx)

                labels.append(label)

                img = osp.join(self.path, random.choice(pair2img[idx]))
                pair_img = osp.join(self.path, random.choice(pair2img[pair_idx]))

                self.pairs.append((img, pair_img, label))

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

        img = cv2.imread(img)
        img_pair = cv2.imread(img_pair)

        if self.aug is not None:
            img = self.aug(image=img)['image']
            img_pair = self.aug(image=img_pair)['image']

        img = to_tensor(img)
        img_pair = to_tensor(img_pair)

        label = torch.Tensor([label])

        return (img, img_pair), label


def get_ds(train_aug=None, valid_aug=None, path='/home/lyan/Documents/fish', encode_labels=False):
    train = pd.read_csv(osp.join(path, 'train.csv'))

    if encode_labels:
        encoder = LabelEncoder()
        train.Id = encoder.fit_transform(train.Id)

    train, valid = train_test_split(train)
    train.reset_index(inplace=True)
    valid.reset_index(inplace=True)

    train_ds = SeameseFishDs(train, osp.join(path, 'train'), train_aug)
    valid_ds = SeameseFishDs(valid, osp.join(path, 'train'), valid_aug)

    return train_ds, valid_ds