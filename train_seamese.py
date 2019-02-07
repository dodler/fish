import collections

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
torch.cuda.set_device(0)
print(torch.cuda.device_count())
from catalyst.dl.callbacks import (
    LossCallback,
    Logger, TensorboardLogger,
    OptimizerCallback, CheckpointCallback,
    OneCycleLR)
from torch.utils.data import DataLoader

from data.ds_factory import get_siamese_ds_no_new_whale_label_encode
from losses import ContrastiveLoss
from models import get_model
from train_utils import SeameseRunner
from utils import get_aug

train_aug, valid_aug = get_aug()

train_ds, valid_ds = get_siamese_ds_no_new_whale_label_encode(train_aug, valid_aug)

train_loader = DataLoader(train_ds, shuffle=True, num_workers=10, pin_memory=True, batch_size=128)
valid_loader = DataLoader(valid_ds, shuffle=False, num_workers=10, pin_memory=True, batch_size=128)

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader

model = get_model('seamese_se_resnext50')

criterion = ContrastiveLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)


# the only tricky part
n_epochs = 3
logdir = "/tmp/runs/"

callbacks = collections.OrderedDict()

callbacks["loss"] = LossCallback()
callbacks["optimizer"] = OptimizerCallback()
# callbacks["map"] = MAPCallback()

# callbacks['show_emb'] = ShowEmbeddingsCallback()
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