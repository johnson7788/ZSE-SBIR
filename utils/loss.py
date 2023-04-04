import torch
import torch.nn as nn


def triplet_loss(x, args):
    """

    :param x: 4*batch -> sk_p, sk_n, im_p, im_n
    :param args:
    :return:
    """
    triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    if not args.cpu:
        triplet = triplet.cuda()
    sk_p = x[0:args.batch]
    im_p = x[2 * args.batch:3 * args.batch]
    im_n = x[3 * args.batch:]
    loss = triplet(sk_p, im_p, im_n)

    return loss


def rn_loss(predict, target, args):
    mse_loss = nn.MSELoss()
    if not args.cpu:
        mse_loss = mse_loss.cuda()
    loss = mse_loss(predict, target)

    return loss


def classify_loss(predict, target, args):
    class_loss = nn.CrossEntropyLoss()
    if not args.cpu:
        class_loss = class_loss.cuda()
    loss = class_loss(predict, target)

    return loss



