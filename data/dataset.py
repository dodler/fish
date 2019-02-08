import os.path as osp
import random
from tqdm import *
import cv2
import numpy as np
import pandas as pd
import torch

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


class SiameseFishDs:
    def __init__(self, images, labels, path, aug, positive_pair_num=2, negative_pair_num=2):
        self.labels = labels
        self.positive_pair_num = positive_pair_num
        self.negative_pair_num = negative_pair_num
        self.images = images
        self.path = path
        self.aug = aug

        self.pairs = []
        self.make_pairs()

    def make_pairs(self):

        pair2img = {}
        for i in tqdm(range(len(self.images))):
            idx = self.labels[i]
            if idx not in pair2img.keys():
                pair2img[idx] = []

            pair2img[idx].append(self.images[i])

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
                labels.append(0.0)
                img = osp.join(self.path, random.choice(pair2img[idx]))
                pair_img = osp.join(self.path, random.choice(pair2img[pair_idx]))

                self.pairs.append((img, pair_img, 0.0))

        print(np.histogram(labels, bins=3))
        print('pairs done,', self.pairs[0:5], ',')

    def __len__(self):
        return len(self.pairs)

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
    def __init__(self, images, labels, path, aug):
        self.labels = labels
        self.images = images
        self.path = path
        self.aug = aug

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        # item = random.randrange(0, self.images.shape[0])

        img_path = osp.join(self.path, self.images[item])
        img = read_img(img_path)

        if self.aug is not None:
            img = self.aug(image=img)['image']

        img = to_tensor(img)

        label = torch.from_numpy(np.array([self.labels[item]])).long()
        return img, label


