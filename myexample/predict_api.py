import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from options import Option
from model.model import Model
from utils.util import setup_seed, load_checkpoint
from utils.ap import calculate
from data_utils.utils import preprocess
from data_utils.preLoad import PreLoad

class ValidSet(torch.utils.data.Dataset):

    def __init__(self, pre_load, type_skim='im', half=False, path=False):
        self.type_skim = type_skim
        self.half = half
        self.path = path
        if type_skim == "sk":
            self.file_names, self.cls = pre_load.all_valid_or_test_sketch, pre_load.all_valid_or_test_sketch_label
        elif type_skim == "im":
            # all_valid_or_test_image: 所有的验证集或者测试集图片, all_valid_or_test_image_label: 所有的验证集或者测试集图片的标签id
            self.file_names, self.cls = pre_load.all_valid_or_test_image, pre_load.all_valid_or_test_image_label
        elif type_skim == "train_sketch":
            self.file_names, self.cls, self.label_names = pre_load.all_train_sketch, pre_load.all_train_sketch_label, pre_load.all_train_sketch_cls_name
        else:
            NameError(type_skim + " is not right")


    def __getitem__(self, index):
        # index： 一条数据的索引
        label = self.cls[index]  # label 为数字
        file_name = self.file_names[index]
        if self.path:
            image = file_name
        else:
            # 对图片预处理
            if self.half:
                image = preprocess(file_name, self.type_skim).half()
            else:
                image = preprocess(file_name, self.type_skim)
        return image, label

    def __len__(self):
        return len(self.file_names)
def valid_cls(args, model, sk_valid_data, im_valid_data):
    """评估数据集"""
    model.eval()
    torch.set_grad_enabled(False)

    print('加载普通图像数据')
    sk_dataload = DataLoader(sk_valid_data, batch_size=args.test_sk, num_workers=args.num_workers, drop_last=False)
    print('加载草图数据')
    im_dataload = DataLoader(im_valid_data, batch_size=args.test_im, num_workers=args.num_workers, drop_last=False)

    dist_im = None
    all_dist = None
    assert len(sk_dataload) >0, "sketch data是空的，这是有问题的"
    for i, (sk, sk_label) in enumerate(tqdm(sk_dataload, desc="草图")):
        if i == 0:
            all_sk_label = sk_label.numpy() #所有草图的标签
        else:
            all_sk_label = np.concatenate((all_sk_label, sk_label.numpy()), axis=0)

        sk_len = sk.size(0) #批次大小，eg:20
        if not args.cpu:
            sk = sk.cuda()
        #sk_idxs,keep_rate小于1的时候，才有用，否则返回None
        sk, sk_idxs = model(sk, None, 'test', only_sa=True)  #根据草图，只计算自注意力，sk:[20,197,768]代表[batch_size,seq_length,hidden_size],sk_idxs:12

        for j, (im, im_label) in enumerate(tqdm(im_dataload, desc="普通图像")):
            if i == 0 and j == 0:
                all_im_label = im_label.numpy()
            elif i == 0 and j > 0:
                all_im_label = np.concatenate((all_im_label, im_label.numpy()), axis=0)

            im_len = im.size(0)
            if not args.cpu:
                im = im.cuda()
            im, im_idxs = model(im, None, 'test', only_sa=True) #根据普通图像，只计算自注意力，im:[20,197,768]代表[batch_size,seq_length,hidden_size],im_idxs:12
            #sk:[20,197,768]代表[batch_size,seq_length,hidden_size], im_len:20,代表图像的个数，sk代表草图,-->[400,197,768]
            sk_temp = sk.unsqueeze(1).repeat(1, im_len, 1, 1).flatten(0, 1)
            if not args.cpu:
                sk_temp = sk_temp.cuda()
            im_temp = im.unsqueeze(0).repeat(sk_len, 1, 1, 1).flatten(0, 1) #普通图像进行扩充，im_temp:[400,197,768]
            if not args.cpu:
                im_temp = im_temp.cuda()

            if args.retrieval == 'rn':
                feature_1, feature_2 = model(sk_temp, im_temp, 'test')  #使用核关系网络, feature_1:代表cls特征,[800,768],feature_2:核关系网络的预测结果,[400,1]
            if args.retrieval == 'sa':
                feature_1, feature_2 = torch.cat((sk_temp[:, 0], im_temp[:, 0]), dim=0), None

            # print(feature_1.size())    # [2*sk*im, 768]
            # print(feature_2.size())    # [sk*im, 1]

            if args.retrieval == 'rn':
                if j == 0:
                    # 距离，sk_len:20, im_len:20, feature_2:[400,1], dist_im:[20,20]，使用view(sk_len, im_len)方法将feature_2张量重新形状为一个[sk_len, im_len]的矩阵。然后，使用cpu()方法将该矩阵移动到CPU上，并使用data.numpy()方法将其转换为一个NumPy数组。最后，使用负号将数组中的每个元素取反，得到一个形状为[sk_len, im_len]的矩阵dist_im。
                    dist_im = - feature_2.view(sk_len, im_len).cpu().data.numpy()  # 1*args.batch, [20,20]
                else:
                    dist_im = np.concatenate((dist_im, - feature_2.view(sk_len, im_len).cpu().data.numpy()), axis=1)
            if args.retrieval == 'sa':
                dist_temp = F.pairwise_distance(F.normalize(feature_1[:sk_len * im_len]),
                                                F.normalize(feature_1[sk_len * im_len:]), 2)
                if j == 0:
                    dist_im = dist_temp.view(sk_len, im_len).cpu().data.numpy()
                else:
                    dist_im = np.concatenate((dist_im, dist_temp.view(sk_len, im_len).cpu().data.numpy()), axis=1)

        if i == 0:
            all_dist = dist_im
        else:
            all_dist = np.concatenate((all_dist, dist_im), axis=0)

    # print(all_sk_label.size, all_im_label.size)     # [762 x 1711] / 2， all_sk_label:[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],numpy array
    #将所有草图和图像的类别进行比较，首先使用np.expand_dims函数将all_sk_label和all_im_label转换为形状为(num_sketches, 1)和(1, num_images)的矩阵。然后，使用==运算符将这两个矩阵逐元素比较，得到一个形状为(num_sketches, num_images)的布尔矩阵。最后，使用* 1运算符将布尔矩阵转换为数值矩阵，其中1表示相同类别，0表示不同类别。
    class_same = (np.expand_dims(all_sk_label, axis=1) == np.expand_dims(all_im_label, axis=0)) * 1
    # print(all_dist.size, class_same.size)     # [762 x 1711] / 2
    map_all, map_200, precision100, precision200 = calculate(all_dist, class_same, test=True)

    return map_all, map_200, precision100, precision200

def load_data_test(args):
    #加载所有数据集
    pre_load = PreLoad(args)
    if args.cpu:
        half = False
    else:
        half = True
    #从所有数据集中选择草图和普通数据集
    train_sketch = ValidSet(pre_load, 'train_sketch', half=half)
    return train_sketch

def test():
    #加载草图和普通图像数据集
    sk_valid_data, im_valid_data = load_data_test(args)

    # prepare model
    model = Model(args)
    if not args.cpu:
        model = model.half()

    if args.load is not None:
        assert os.path.isfile(args.load), f'错误: 没有找到模型,请检查路径!, {args.load}'
        checkpoint = load_checkpoint(args.load, args.cpu)
    cur = model.state_dict()
    new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
    cur.update(new)
    model.load_state_dict(cur)

    if len(args.choose_cuda) > 1:
        model = torch.nn.parallel.DataParallel(model.to('cuda'))
    if not args.cpu:
        model = model.cuda()

    # valid
    map_all, map_200, precision_100, precision_200 = valid_cls(args, model, sk_valid_data, im_valid_data)
    print(f'map_all:{map_all:.4f} map_200:{map_200:.4f} precision_100:{precision_100:.4f} precision_200:{precision_200:.4f}')

if __name__ == '__main__':
    args = Option().parse()
    print("test args:", str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)
    test()
