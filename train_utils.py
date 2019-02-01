import torch

from catalyst.dl.callbacks import MultiMetricCallback, metrics, Callback, UtilsFactory
from catalyst.dl.metrics import average_precision
from catalyst.dl.runner import BaseModelRunner
from typing import Dict, List
import torch.nn as nn
import numpy as np
from catalyst.dl.state import RunnerState
from torch import optim


class SeameseRunner(BaseModelRunner):

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module = None,
                 optimizer: optim.Optimizer = None,
                 scheduler: optim.lr_scheduler._LRScheduler = None):
        super(SeameseRunner, self).__init__(model, criterion, optimizer, scheduler)

    def batch2device(self, *, dct: Dict, state: RunnerState = None):

        if isinstance(dct, (tuple, list)):
            assert len(dct) == 2
            dct = {"features": dct[0], "targets": dct[1]}

        if state is not None:
            dct = {
                "features": [
                    dct["features"][0].to(0),
                    dct["features"][1].to(0)
                ],
                "targets": dct['targets'].to(0)
            }
        else:
            dct = {key: value.to(self.device) for key, value in dct.items()}

        return dct

    def batch_handler(
            self, *, dct: Dict, model: nn.Module, state: RunnerState = None
    ) -> Dict:
        """
        Batch handler wrapper with main statistics and device management.

        :param dct: key-value storage with input tensors
        :param model: model to predict with
        :param state: runner state
        :return: key-value storage with model predictions
        """
        dct = self.batch2device(dct=dct, state=state)
        logits = model(dct["features"])
        output = {"logits": logits}

        return output


class MAPCallback(MultiMetricCallback):
    """
    Precision metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            precision_args: List[int] = None,
            prefix="map"
    ):
        super().__init__(
            prefix=prefix,
            metric_fn=mean_average_precision,
            list_args=precision_args or [5],
            input_key=input_key,
            output_key=output_key
        )


def mean_average_precision(outputs, targets, topk=(1,)):
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two lists
        of lists of items.
    Parameters
    ----------
    outputs : list
                A list of lists of predicted elements
    targets : list
             A list of lists of elements that are to be predicted
    topk : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    max_k = max(topk)
    _, pred = outputs.topk(max_k, 1, True, True)

    targets = targets.data.cpu().numpy().tolist()
    actual_list = []
    for a in targets:
        actual_list.append([a])
    targets = actual_list
    pred = pred.tolist()

    res = []
    for k in topk:
        ap = np.mean(
            [average_precision(p, a, k) for a, p in zip(targets, pred)]
        )
        res.append(ap)
    return res