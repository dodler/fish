import os
import time

from sklearn.neighbors import NearestNeighbors
from tqdm import *
from tensorboardX import SummaryWriter
from args import get_parser

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

torch.cuda.set_device(0)
print(torch.cuda.device_count())
from torch.utils.data import DataLoader

from dataset import get_seamese_ds, get_classif_ds
from losses import ContrastiveLoss
from models import get_model
from utils import get_aug, map_per_set
import numpy as np

DISP_NUM = 8

parser = get_parser()
args = parser.parse_args()

train_aug, valid_aug = get_aug()

train_ds, valid_ds, encoder = get_seamese_ds(train_aug, valid_aug, drop_new_whale=False, encode_labels=True)

train_loader = DataLoader(train_ds, shuffle=True, num_workers=10, pin_memory=True, batch_size=args.batch_size)
valid_loader = DataLoader(valid_ds, shuffle=False, num_workers=10, pin_memory=True, batch_size=args.batch_size)

model = get_model(args.name)
model.to(0)
model.train()

criterion = ContrastiveLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

writer = SummaryWriter(log_dir='/tmp/runs_new')

stp = 0


def print_loss(loss, accu, out1, out2, labels):
    global stp
    writer.add_scalar('loss', loss, global_step=stp)
    writer.add_scalar('accu', accu, global_step=stp)

    r = ''
    for i in range(DISP_NUM):
        r += ('d:' + str(np.linalg.norm((out1[i] - out2[i]).detach().cpu().numpy())) + ', l:' +
              str(labels[i].detach().cpu().numpy()))

    writer.add_text('dist', r, stp)

    writer.add_histogram('labels', labels)
    stp += 1


def train(epoch):
    train_loss = []
    model.train()
    start = time.time()
    start_epoch = time.time()
    for batch_idx, (x, labels) in enumerate(train_loader):
        labels = labels.float()

        x0, x1 = x

        x0, x1, labels = x0.cuda(), x1.cuda(), labels.cuda()

        out = model((x0, x1))
        output1, output2 = out
        loss = criterion((output1, output2), labels)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = []

        for idx, logit in enumerate([output1, output2]):
            corrects = (torch.max(logit, 1)[1].data == labels.long().data).sum()
            accu = float(corrects) / float(labels.size()[0])
            accuracy.append(accu)

        print_loss(loss, accu, output1, output2, labels)

        if batch_idx % 23 == 0:
            end = time.time()
            took = end - start

            writer.add_images('wh1', 0.5 + x0[0:DISP_NUM, :, :, :].squeeze())
            writer.add_images('wh2', 0.5 + x1[0:DISP_NUM, :, :, :].squeeze())

            for idx, accu in enumerate(accuracy):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}\tTook: {:.2f}\tOut: {}\tAccu: {:.2f}'.format(
                    epoch, batch_idx * len(labels), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                    took, idx, accu * 100.))
            start = time.time()
    torch.save(model.state_dict(), f'{args.name}_{epoch}_{loss}.pth')
    end = time.time()
    took = end - start_epoch
    print('Train epoch: {} \tTook:{:.2f}'.format(epoch, took))
    return train_loss


def validate():
    model.eval()

    print('calculating embeddings')

    _, aug = get_aug()

    cls_train_ds, cls_valid_ds, enc = get_classif_ds(aug, aug, drop_new_whale=False, encode_labels=True)

    cls_valid_loader = DataLoader(cls_valid_ds, shuffle=False, num_workers=20, pin_memory=True, batch_size=128)
    cls_train_loader = DataLoader(cls_train_ds, shuffle=False, num_workers=20, pin_memory=True, batch_size=128)

    train_embs = []
    for img, label in tqdm(cls_train_loader):
        with torch.no_grad():
            img = img.to(0)
            train_embs.append(model.embedding(img))

    valid_embs = []
    for img, label in tqdm(cls_valid_loader):
        with torch.no_grad():
            img = img.to(0)
            valid_embs.append(model.embedding(img))

    train_embs = np.concatenate([k.detach().cpu().numpy() for k in train_embs])
    valid_embs = np.concatenate([k.detach().cpu().numpy() for k in valid_embs])

    gt = enc.inverse_transform(cls_valid_ds.df.Id)
    print('fitting knn')
    knn = NearestNeighbors(n_neighbors=16)
    knn.fit(train_embs, cls_train_ds.df.Id)

    nbrs_5 = knn.kneighbors(valid_embs)
    predictions = []
    for i in tqdm(range(cls_valid_ds.df.shape[0])):
        labels, counts = np.unique(cls_train_ds.df.Id[nbrs_5[1][i]].values, return_counts=True)
        labels = labels[np.argsort(counts)[::-1][:5]]

        predictions.append(enc.inverse_transform(labels).tolist())

    print(map_per_set(gt, predictions))


validate()
for i in range(args.epochs):
    train(i)
    if i % 3 == 0:
        validate()
