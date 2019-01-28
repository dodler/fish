from torch import nn
from torchvision.models import resnet34


def get_model(model_name='resnet34'):
    return Resnet34Seamese()


class Resnet34Seamese(nn.Module):

    def __init__(self):
        super(Resnet34Seamese, self).__init__()

        self.model = resnet34(pretrained='imagenet')
        self.model.fc = nn.Linear(512, 128)

    def forward(self, img_pair):
        return (self.model(img_pair[0]), \
               self.model(img_pair[1]))
