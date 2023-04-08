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

class SketchSet(torch.utils.data.Dataset):

    def __init__(self, name2id,file_ls_file, type_skim='sketch', half=False, path=False, mini=True):
        """

        """
        image_dir = "datasets/comestic"
        self.type_skim = type_skim
        self.half = half
        self.path = path
        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()
        # 图片相对路径
        file_list = [os.path.join(image_dir,f) for f in file_content]
        file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_list])
        # 图片的label,0,1,2...
        labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])

        # 所有的训练类
        with open(name2id, 'r') as ci:
            class_and_indx = ci.readlines()
        # 类名
        cname = np.array([' '.join(cc.strip().split()[:-1]) for cc in class_and_indx])
        # file_ls:所有的图片路径, labels:所有的标签,是数字的array, cname:所有的标签名称
        if mini:
            self.file_names, self.cls, self.label_names = file_ls[:100], labels[:100], cname
        else:
            self.file_names, self.cls, self.label_names = file_ls, labels, cname


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


class PredictSet(torch.utils.data.Dataset):
    def __init__(self, image_dir, half=False):
        """
        image_dir: 图片路径
        """
        self.half = half
        self.path = image_dir
        self.type_skim = "im"  # 代表处理image，而不是sketch
        # 要预测的图片目录
        images = os.listdir(image_dir)
        self.file_names = [os.path.join(image_dir, image) for image in images]
        self.cls = [0] * len(self.file_names)

    def __getitem__(self, index):
        # index： 一条数据的索引
        label = self.cls[index]  # label 为数字
        file_name = self.file_names[index]
        # 对图片预处理
        if self.half:
            image = preprocess(file_name, self.type_skim).half()
        else:
            image = preprocess(file_name, self.type_skim)
        return image, label

    def __len__(self):
        return len(self.file_names)

def predict(image_dir):
    #加载草图和普通图像数据集
    if args.cpu:
        half = False
    else:
        half = True
    cname_cid = 'datasets/comestic/zeroshot/cname_cid_zero.txt'
    file_ls_file = 'datasets/comestic/zeroshot/all_photo_filelist_zero.txt'
    image_dataset = SketchSet(half=half,name2id=cname_cid, file_ls_file=file_ls_file, type_skim="im",mini=False)
    # image_dataset = PredictSet(image_dir, half=half)
    cname_cid = 'datasets/comestic/zeroshot/cname_cid.txt'
    file_ls_file = 'datasets/comestic/zeroshot/sketch_tx_000000000000_ready_filelist_train.txt'
    train_sketch_dataset = SketchSet(half=half,name2id=cname_cid, file_ls_file=file_ls_file, mini=False)
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
    valid_cls(args, model, train_sketch_dataset,image_dataset)

def valid_cls(args, model, train_sketch_dataset,image_dataset):
    """评估数据集"""
    model.eval()
    torch.set_grad_enabled(False)

    print('加载普通图像数据')
    train_sketch_dataload = DataLoader(train_sketch_dataset, batch_size=args.test_sk, num_workers=args.num_workers, drop_last=False)
    image_dataload = DataLoader(image_dataset, batch_size=args.test_sk, num_workers=args.num_workers, drop_last=False)
    dist_im = None
    all_dist = None
    assert len(image_dataload) >0, "要预测的image_dataload是空的，这是有问题的"
    sk_embedding = []
    sk_batches = []
    for i, (sk, sk_label) in enumerate(tqdm(train_sketch_dataload, desc="草图")):
        if i == 0:
            all_sk_label = sk_label.numpy()  # 所有草图的标签
        else:
            all_sk_label = np.concatenate((all_sk_label, sk_label.numpy()), axis=0)

        sk_len = sk.size(0)  # 批次大小，eg:20
        if not args.cpu:
            sk = sk.cuda()
        # sk_idxs,keep_rate小于1的时候，才有用，否则返回None
        sk, _ = model(sk, None, 'test',
                            only_sa=True)  # 根据草图，只计算自注意力，sk:[20,197,768]代表[batch_size,seq_length,hidden_size],sk_idxs:12
        sk_embedding.append(sk)
        sk_batches.append(sk_len)
    im_embedding = []
    im_batches = []
    for j, (im, im_label) in enumerate(tqdm(image_dataload, desc="普通图像")):
            if j == 0:
                all_im_label = im_label.numpy()
            elif j > 0:
                all_im_label = np.concatenate((all_im_label, im_label.numpy()), axis=0)
            im_len = im.size(0)
            if not args.cpu:
                im = im.cuda()
            im, im_idxs = model(im, None, 'test',only_sa=True)  # 根据普通图像，只计算自注意力，im:[20,197,768]代表[batch_size,seq_length,hidden_size],im_idxs:12
            im_embedding.append(im)
            im_batches.append(im_len)

    for sk_index, (sk, sk_len) in enumerate(zip(sk_embedding, sk_batches)):
        for im_index,(im, im_len) in enumerate(zip(im_embedding, im_batches)):
            # sk:[20,197,768]代表[batch_size,seq_length,hidden_size], im_len:20,代表图像的个数，sk代表草图,-->[400,197,768]
            sk_temp = sk.unsqueeze(1).repeat(1, im_len, 1, 1).flatten(0, 1)
            if not args.cpu:
                sk_temp = sk_temp.cuda()
            im_temp = im.unsqueeze(0).repeat(sk_len, 1, 1, 1).flatten(0, 1)  # 普通图像进行扩充，im_temp:[400,197,768]
            if not args.cpu:
                im_temp = im_temp.cuda()

            feature_1, feature_2 = model(sk_temp, im_temp,
                                         'test')  # 使用核关系网络, feature_1:代表cls特征,[800,768],feature_2:核关系网络的预测结果,[400,1]
            if im_index == 0:
                # 距离，sk_len:20, im_len:20, feature_2:[400,1], dist_im:[20,20]，使用view(sk_len, im_len)方法将feature_2张量重新形状为一个[sk_len, im_len]的矩阵。然后，使用cpu()方法将该矩阵移动到CPU上，并使用data.numpy()方法将其转换为一个NumPy数组。最后，使用负号将数组中的每个元素取反，得到一个形状为[sk_len, im_len]的矩阵dist_im。
                dist_im = - feature_2.view(sk_len, im_len).cpu().data.numpy()  # 1*args.batch, [20,20]
            else:
                dist_im = np.concatenate((dist_im, - feature_2.view(sk_len, im_len).cpu().data.numpy()), axis=1)
        if sk_index == 0:
            all_dist = dist_im
        else:
            all_dist = np.concatenate((all_dist, dist_im), axis=0)

    # print(all_sk_label.size, all_im_label.size)     # [762 x 1711] / 2， all_sk_label:[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],numpy array
    #将所有草图和图像的类别进行比较，首先使用np.expand_dims函数将all_sk_label和all_im_label转换为形状为(num_sketches, 1)和(1, num_images)的矩阵。然后，使用==运算符将这两个矩阵逐元素比较，得到一个形状为(num_sketches, num_images)的布尔矩阵。最后，使用* 1运算符将布尔矩阵转换为数值矩阵，其中1表示相同类别，0表示不同类别。
    class_same = (np.expand_dims(all_sk_label, axis=1) == np.expand_dims(all_im_label, axis=0)) * 1
    # print(all_dist.size, class_same.size)     # [762 x 1711] / 2
    accuracy = calculate_acc(all_dist, class_same)
    print(accuracy)
    return accuracy

def calculate_acc(distance, class_same):
    """
    计算模型的准确率，只考虑最近距离的样本，distance:距离矩阵,[库中所有商品数量,要预测的商品数量]，class_same:标签矩阵,[库中所有商品数量,要预测的商品数量]
    """
    arg_sort_sim = distance.argsort()   # 得到从小到大索引值， 所有距离
    sort_label = []
    for index in range(0, arg_sort_sim.shape[0]):
        # 将label重新排序，根据距离的远近，距离越近的排在前面
        sort_label.append(class_same[index, arg_sort_sim[index, :]])
    sort_label = np.array(sort_label)

    # 只考虑最近距离的样本
    nearest_label = sort_label[:, 0]

    # 计算准确率
    accuracy = np.mean(nearest_label)

    return accuracy

if __name__ == '__main__':
    args = Option().parse()
    print("test args:", str(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.choose_cuda
    print("current cuda: " + args.choose_cuda)
    setup_seed(args.seed)
    predict(image_dir="datasets/comestic/test_image")
