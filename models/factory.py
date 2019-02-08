from models.base_models import  *
from models.new_classification import get_new_non_new_se_resnext50


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
    elif model_name == 'new_classification_se_resnext50':
        return get_new_non_new_se_resnext50()