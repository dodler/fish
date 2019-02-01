import os
import time

from tensorboardX import SummaryWriter

from args import get_parser

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

torch.cuda.set_device(0)
print(torch.cuda.device_count())
from torch.utils.data import DataLoader

from dataset import get_seamese_ds
from losses import ContrastiveLoss
from models import get_model
from utils import get_aug
import numpy as np

DISP_NUM = 8

parser = get_parser()
args = parser.parse_args()

train_aug, valid_aug = get_aug()

train_ds, valid_ds = get_seamese_ds(train_aug, valid_aug, drop_new_whale=False)

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
        r += ('d:' + str(np.linalg.norm((out1[i]-out2[i]).detach().cpu().numpy())) + ', l:' +
                        str(labels[i].detach().cpu().numpy()))

    writer.add_text('dist', r, stp)

    writer.add_histogram('labels',labels)
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

            writer.add_images('wh1', 0.5+x0[0:DISP_NUM, :, :, :].squeeze())
            writer.add_images('wh2', 0.5+x1[0:DISP_NUM, :, :, :].squeeze())

            for idx, accu in enumerate(accuracy):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}\tTook: {:.2f}\tOut: {}\tAccu: {:.2f}'.format(
                    epoch, batch_idx * len(labels), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                    took, idx, accu * 100.))
            start = time.time()
    torch.save(model.state_dict(), './model-epoch-%s.pth' % epoch)
    end = time.time()
    took = end - start_epoch
    print('Train epoch: {} \tTook:{:.2f}'.format(epoch, took))
    return train_loss


def test(model):
    model.eval()
    all = []
    all_labels = []

    for batch_idx, (x, labels) in enumerate(valid_loader):
        x, labels = x.cuda(), labels.cuda()

        output = model.embedding(x)
        all.extend(output.data.cpu().numpy().tolist())
        all_labels.extend(labels.data.cpu().numpy().tolist())

    numpy_all = np.array(all)
    numpy_labels = np.array(all_labels)
    return numpy_all, numpy_labels


for i in range(args.epochs):
    train(i)

test(model)
