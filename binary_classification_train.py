import os
import time

from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import *
from tensorboardX import SummaryWriter
from args import get_parser

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

torch.cuda.set_device(0)
print(torch.cuda.device_count())
from torch.utils.data import DataLoader

from data.ds_factory import get_classification_ds_with_new_whale, get_siamese_ds_no_new_whale_label_encode
from losses import ContrastiveLoss
from models import get_model
from utils import get_aug, map_per_set
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
    encoder_type='one_hot')

train_loader = DataLoader(train_ds, shuffle=True, num_workers=10, pin_memory=True, batch_size=args.batch_size)
valid_loader = DataLoader(valid_ds, shuffle=False, num_workers=10, pin_memory=True, batch_size=args.batch_size)

model = get_model(args.name)
model.to(0)
model.train()

criterion = BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=5e-3)
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


def train(epoch):
    global stp

    train_loss = []
    model.train()
    start = time.time()
    start_epoch = time.time()
    it = tqdm(enumerate(train_loader))
    for batch_idx, (x, labels) in it:
        labels = labels.float()

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

        pred = (torch.sigmoid(output).squeeze().detach().cpu().numpy() > 0.5).astype(np.int)
        gt = labels.squeeze().detach().cpu().numpy()

        pred = encoder.inverse_transform(pred)
        gt = encoder.inverse_transform(gt)

        score = accuracy_score(pred, gt)
        print_data(loss, score, labels)

        end = time.time()
        took = end - start

        writer.add_images('wh1', 0.5 + x[0:DISP_NUM, :, :, :].squeeze())

        it.set_description_str(f'train score {score}, loss {round(loss.item(),3)} gt {gt[0]}, pred {str(pred[0:2])}')
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
        labels = labels.float()

        x, labels = x.cuda(), labels.cuda()

        with torch.no_grad():
            out = model(x)

        output = out
        loss = criterion(output.squeeze(), labels.squeeze())

        valid_loss.append(loss.item())
        optimizer.zero_grad()

        pred = (torch.sigmoid(output).squeeze().detach().cpu().numpy() > 0.5).astype(np.int)
        gt = labels.squeeze().detach().cpu().numpy()

        pred = encoder.inverse_transform(pred)
        gt = encoder.inverse_transform(gt)

        score = accuracy_score(pred, gt)
        it.set_description_str(f'valid score {score}, valid loss {round(loss.item(),3)} gt {gt[0]}, pred {str(pred[0:2])})')

    torch.save(model.state_dict(), osp.join(checkpoints_directory, f'{args.name}_{epoch}_{loss}.pth'))
    end = time.time()
    took = end - start_epoch
    print('Valid epoch: {} \tTook:{:.2f}'.format(epoch, took))
    return valid_loss


for i in range(args.epochs):
    train(i)
    if i % 3 == 0:
        validate(i)
