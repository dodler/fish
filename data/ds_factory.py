from os import path as osp

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from data.dataset import FishDs, SiameseFishDs


def get_classification_ds_with_new_whale(train_aug=None, valid_aug=None,
                                         path='/home/lyan/Documents/fish',
                                         encode_labels=True,
                                         encoder_type='label'):
    train = pd.read_csv(osp.join(path, 'train.csv'))

    # print('shape before dropping', train.Id.shape)
    # train = train[train.Id != 'new_whale']
    train_images = train.Image.values
    train_labels = train.Id.values
    # print('shape after dropping', train_images.shape)

    if encode_labels:
        if encoder_type == 'one_hot':
            encoder = OneHotEncoder()
            train_labels = encoder.fit_transform(train_labels.reshape(-1, 1)).todense()
        elif encoder_type == 'label':
            encoder = LabelEncoder()
            train_labels = encoder.fit_transform(train_labels)
        else:
            raise Exception('supported only one_hot and label encoders')

    train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels)

    print(train_labels.shape, valid_labels.shape)

    train_ds = FishDs(train_images, train_labels, osp.join(path, 'train'), train_aug)
    valid_ds = FishDs(valid_images, valid_labels, osp.join(path, 'train'), valid_aug)

    if encode_labels:
        return train_ds, valid_ds, encoder
    else:
        return train_ds, valid_ds


def get_siamese_ds_no_new_whale_label_encode(train_aug=None,
                                             valid_aug=None,
                                             path='/home/lyan/Documents/fish',
                                             encode_labels=False):
    train = pd.read_csv(osp.join(path, 'train.csv'))
    train = train[train.Id != 'new_whale']

    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train.Id)
    train_images = train.Image.values

    train_ds, valid_ds = siamese_ds_from_df(path, train_images, train_labels, train_aug, valid_aug)

    if encode_labels:
        return train_ds, valid_ds, encoder
    else:
        return train_ds, valid_ds


def get_siamese_ds_no_new_whale_raw_labels(train_aug=None,
                                           valid_aug=None,
                                           path='/home/lyan/Documents/fish'):
    train = pd.read_csv(osp.join(path, 'train.csv'))
    train = train[train.Id != 'new_whale']

    train_labels = train.Id.values
    train_images = train.Image

    train_ds, valid_ds = siamese_ds_from_df(path, train_images, train_labels, train_aug, valid_aug)

    return train_ds, valid_ds


def siamese_ds_from_df(path, train_images, train_labels, train_aug, valid_aug):
    train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels)

    train_ds = SiameseFishDs(train_images, train_labels, osp.join(path, 'train'), train_aug)
    valid_ds = SiameseFishDs(valid_images, valid_labels, osp.join(path, 'train'), valid_aug)
    return train_ds, valid_ds
