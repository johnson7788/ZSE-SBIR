import os
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import Option
from data_utils.dataset import load_data
from model.model import Model
from utils.util import build_optimizer, save_checkpoint, setup_seed
from utils.loss import triplet_loss, rn_loss
from utils.valid import valid_cls


def train():
    train_data, sk_valid_data, im_valid_data = load_data(args)

    model = Model(args)
    if not args.cpu:
        model = model.cuda()

    # batch=15, lr=1e-5 / batch=30, lr=2e-5
    optimizer = build_optimizer(args, model)

    train_data_loader = DataLoader(train_data, args.batch, num_workers=2, drop_last=True)

    start_epoch = 0
    accuracy = 0

    for i in range(start_epoch, args.epoch):
        print('------------------------train------------------------')
        epoch = i + 1
        model.train()
        torch.set_grad_enabled(True)

        start_time = time.time()
        num_total_steps = args.datasetLen // args.batch

        for index, (sk, im, sk_neg, im_neg, sk_label, im_label, _, _) in enumerate(train_data_loader):
            print(f"正在训练第 {epoch} 个epoch的第 {index} 个batch")
            sk = torch.cat((sk, sk_neg))  #shape: (30, 3, 224, 224)
            im = torch.cat((im, im_neg))  #shape: (30, 3, 224, 224)
            if not args.cpu:
                sk, im = sk.cuda(), im.cuda()

            # prepare rn truth, target_rn: (30,)
            target_rn = torch.cat((torch.ones(sk_label.size()), torch.zeros(sk_label.size())), dim=0)
            target_rn = torch.clamp(target_rn, 0.01, 0.99).unsqueeze(dim=1) #shape: (30, 1)
            if not args.cpu:
                target_rn = target_rn.cuda()

            # calculate feature
            cls_fea, rn_scores = model(sk, im)

            # loss
            losstri = triplet_loss(cls_fea, args) * 2   # The initial value of losstri should be around 1.00.
            lossrn = rn_loss(rn_scores, target_rn, args) * 4  # The initial value of lossrn should be around 1.00.
            loss = losstri + lossrn

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            step = index + 1
            if step % 30 == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f'Epoch是:{epoch} step是:{step} 当前Epoch训练完成需要时间: {remaining_time} 总loss:{loss.item():.3f} '
                      f'三元组loss:{losstri.item():.3f} 二分类loss:{lossrn.item():.3f}')

        if epoch >= 10:
            print('------------------------valid------------------------')
            # log
            map_all, map_200, precision_100, precision_200 = valid_cls(args, model, sk_valid_data, im_valid_data)
            print(f'map_all:{map_all:.4f} map_200:{map_200:.4f} precision_100:{precision_100:.4f} precision_200:{precision_200:.4f}')
            # save
            if map_all > accuracy:
                accuracy = map_all
                precision = precision_100
                print("Save the BEST {}th model......".format(epoch))
                save_checkpoint(
                    {'model': model.state_dict(), 'epoch': epoch, 'map_all': accuracy, 'precision_100': precision},
                    args.save, f'best_checkpoint')


if __name__ == '__main__':
    args = Option().parse()
    print("train args:", str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)

    train()
