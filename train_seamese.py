import collections

import torch
from catalyst.dl.callbacks import (
    LossCallback,
    Logger, TensorboardLogger,
    OptimizerCallback, SchedulerCallback, CheckpointCallback,
    PrecisionCallback, OneCycleLR)
from torch.utils.data import DataLoader

from dataset import get_ds
from losses import ContrastiveLoss
from models import get_model
from train_utils import SeameseRunner, MAPCallback
from utils import get_aug

train_aug, valid_aug = get_aug()

train_ds, valid_ds = get_ds(train_aug, valid_aug)

train_loader = DataLoader(train_ds, shuffle=True, num_workers=20, pin_memory=True, batch_size=256)
valid_loader = DataLoader(valid_ds, shuffle=False, num_workers=20, pin_memory=True, batch_size=384)

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader

model = get_model('resnet34')

criterion = ContrastiveLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)


# the only tricky part
n_epochs = 20
logdir = "/tmp/runs/"

callbacks = collections.OrderedDict()

callbacks["loss"] = LossCallback()
callbacks["optimizer"] = OptimizerCallback()
# callbacks["map"] = MAPCallback()

# OneCylce custom scheduler callback
callbacks["scheduler"] = OneCycleLR(
    cycle_len=n_epochs,
    div=3, cut_div=4, momentum_range=(0.95, 0.85))
#
# Pytorch scheduler callback
# callbacks["scheduler"] = SchedulerCallback(
#     reduce_metric="map")

callbacks["saver"] = CheckpointCallback()
callbacks["logger"] = Logger()
callbacks["tflogger"] = TensorboardLogger()

runner = SeameseRunner(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler)
runner.train(
    loaders=loaders,
    callbacks=callbacks,
    logdir=logdir,
    epochs=n_epochs, verbose=True)