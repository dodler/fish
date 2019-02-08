from pretrainedmodels.models.senet import se_resnext50_32x4d
from torch import nn


def get_new_non_new_se_resnext50():
    model = se_resnext50_32x4d(pretrained='imagenet')
    model.last_linear = nn.Linear(2048, 1)

    return model