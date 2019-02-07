import bottleneck
import os
import time

from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import *

from args import get_parser

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

torch.cuda.set_device(0)
print(torch.cuda.device_count())
from torch.utils.data import DataLoader

from data.ds_factory import get_classification_ds_with_new_whale
from models import get_model
from utils import get_aug
import numpy as np
import os.path as osp

DISP_NUM = 8

parser = get_parser()
args = parser.parse_args()

checkpoints_directory = osp.join('checkpoints', args.name)
if not osp.exists(checkpoints_directory):
    os.mkdir(checkpoints_directory)

train_aug, valid_aug = get_aug()

train_ds, valid_ds, encoder = get_classification_ds_with_new_whale(
    train_aug,
    valid_aug,
    encode_labels=True,
    encoder_type='label')

train_loader = DataLoader(train_ds, shuffle=True, num_workers=10, pin_memory=True, batch_size=args.batch_size)
valid_loader = DataLoader(valid_ds, shuffle=False, num_workers=10, pin_memory=True, batch_size=args.batch_size)

model = get_model(args.name)
model.to(0)
model.train()

criterion = CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=5e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                       patience=2,
                                                       verbose=True)

lr_scheduler = ReduceLROnPlateau(patience=1000, verbose=True,
                                 optimizer=optimizer)  # i will try to use it for iteration scheduling

writer = SummaryWriter(log_dir=f'/tmp/runs_{args.name}')

stp = 0


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def print_data(loss, score, labels):
    global stp
    writer.add_scalar('loss', loss, global_step=stp)
    writer.add_scalar('lr', get_lr(optimizer), global_step=stp)
    writer.add_scalar('score', score, global_step=stp)

    writer.add_histogram('labels', labels)
    stp += 1


def top_n_indexes(arr, n):
    idx = bottleneck.argpartition(arr, arr.size - n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


def train(epoch):
    global stp

    train_loss = []
    model.train()
    start = time.time()
    start_epoch = time.time()
    it = tqdm(enumerate(train_loader))
    for batch_idx, (x, labels) in it:
        x, labels = x.cuda(), labels.cuda()

        out = model(x)
        output = out
        loss = criterion(output.squeeze(), labels.squeeze())
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = []

        lr_scheduler.step(loss, stp)

        pred = F.softmax(output).squeeze().detach().cpu().numpy()
        gt = labels.squeeze().detach().cpu().numpy()
        pred = np.argmax(pred, axis=1)
        score = accuracy_score(gt, pred)

        print_data(loss, score, labels)

        end = time.time()
        took = end - start

        writer.add_images('wh1', 0.5 + x[0:DISP_NUM, :, :, :].squeeze())

        pp = str(pred[0])
        it.set_description_str(f'train score {score}, loss {round(loss.item(),3)} gt {gt[0]}, pred {pp}')
        start = time.time()
    torch.save(model.state_dict(), osp.join(checkpoints_directory, f'{args.name}_{epoch}_{loss}.pth'))
    end = time.time()
    took = end - start_epoch
    print('Train epoch: {} \tTook:{:.2f}'.format(epoch, took))
    return train_loss


def validate(epoch):
    model.eval()

    valid_loss = []

    start_epoch = time.time()
    it = tqdm(enumerate(train_loader))
    for batch_idx, (x, labels) in it:
        x, labels = x.cuda(), labels.cuda()

        with torch.no_grad():
            out = model(x)

        output = out
        loss = criterion(output.squeeze(), labels.squeeze())

        valid_loss.append(loss.item())
        optimizer.zero_grad()

        pred = F.softmax(output).squeeze().detach().cpu().numpy()
        gt = labels.squeeze().detach().cpu().numpy()
        pred = np.argmax(pred, axis=1)
        score = accuracy_score(gt, pred)
        print_data(loss, score, labels)

        pp = str(pred[0])
        it.set_description_str(f'valid score {score}, valid loss {round(loss.item(),3)} gt {gt[0]}, pred {pp})')

    torch.save(model.state_dict(), osp.join(checkpoints_directory, f'{args.name}_{epoch}_{loss}.pth'))
    end = time.time()
    took = end - start_epoch
    print('Valid epoch: {} \tTook:{:.2f}'.format(epoch, took))
    return valid_loss


for i in range(args.epochs):
    train(i)
    if i % 3 == 0:
        validate(i)
