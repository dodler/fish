from albumentations import Resize, RandomGamma, Rotate, HorizontalFlip, Compose


def get_aug():
    return Compose([
        HorizontalFlip(p=0.7),
        RandomGamma(p=0.7),
        Rotate(p=0.7),
        Resize(width=224, height=224)
    ]), Compose([
        Resize(width=224, height=224)
    ])