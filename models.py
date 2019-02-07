from torch import nn
from torchvision.models import resnet34
from pretrainedmodels.models import se_resnext50_32x4d, se_resnet101


def get_model(model_name='seamese_resnet34'):
    if model_name == 'seamese_resnet34':
        return Resnet34Seamese()
    elif model_name == 'seamese_se_resnext50':
        return SEResnext50Seamese()
    elif model_name == 'classif_resnet34':
        return classification_resnet34()
    elif model_name == 'classif_se_resnet50':
        return classification_se_resnext50()
    elif model_name == 'classif_se_resnet101':
        return classification_se_resnet101(no_new_whale=False)


def classification_se_resnet101(no_new_whale=True):
    model = se_resnet101(pretrained='imagenet')
    model.avg_pool = nn.AvgPool2d(7, stride=1)
    if no_new_whale:
        model.last_linear = nn.Sequential(
            nn.Linear(2048, 5004)
        )
    else:
        model.last_linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 5005)
        )

    return model


def classification_resnet34():
    model = resnet34(pretrained='imagenet')
    model.fc = nn.Linear(512, 5004)
    return model


def classification_se_resnext50():
    model = se_resnext50_32x4d(pretrained='imagenet')
    model.last_linear = nn.Linear(2048, 5005)

    return model


class Resnet34Seamese(nn.Module):

    def __init__(self):
        super(Resnet34Seamese, self).__init__()

        self.model = resnet34(pretrained='imagenet')
        self.model.fc = nn.Linear(512, 256)

    def forward(self, img_pair):
        return (self.model(img_pair[0]), \
                self.model(img_pair[1]))

    def embedding(self, img):
        return self.model(img)


class SEResnext50Seamese(nn.Module):
    def __init__(self):
        super(SEResnext50Seamese, self).__init__()

        self.model = se_resnext50_32x4d(pretrained='imagenet')
        self.model.last_linear = nn.Linear(2048, 256)

    def forward(self, img_pair):
        return (self.model(img_pair[0]), self.model(img_pair[1]))

    def embedding(self, img):
        return self.model(img)
