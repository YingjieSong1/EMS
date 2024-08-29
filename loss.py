import torch


def loss_func(pred, gt):
    loss =torch.nn.BCELoss()(pred.float(), gt.float())
    return loss


