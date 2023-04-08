import torch
import torch.nn as nn
from .sa import Self_Attention
from .ca import Cross_Attention
from .rn import Relation_Network, cos_similar

from options import Option

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

        self.sa = Self_Attention(d_model=args.d_model, cls_number=args.cls_number, pretrained=args.pretrained)
        self.ca = Cross_Attention(args=args, h=args.head, n=args.number, d_model=args.d_model, d_ff=args.d_ff, dropout=0.1)
        self.rn = Relation_Network(args.anchor_number, dropout=0.1)
        self.conv2d = nn.Conv2d(768, 512, 2, 2)


    def forward(self, sk, im, stage='train', only_sa=False):
        # sk: [b, 3, 224, 224], im: [b, 3, 224, 224]
        if stage == 'train':

            sk_im = torch.cat((sk, im), dim=0) # [2b, 3, 224, 224]
            sa_fea, left_tokens, idxs = self.sa(sk_im)  # self attention  [4b, 197, 768]
            ca_fea = self.ca(sa_fea)  #cross-attention [4b, 197, 768]-->[4b, 197, 768]

            cls_fea = ca_fea[:, 0]  # [4b, 1, 768]
            token_fea = ca_fea[:, 1:]  # [4b, 196, 768]
            batch = token_fea.size(0)

            token_fea = token_fea.view(batch, 768, 14, 14)
            down_fea = self.conv2d(token_fea)
            down_fea = down_fea.view(batch, 512, 7*7)
            down_fea = down_fea.transpose(1, 2)  # [4b, 49, 512]

            sk_fea = down_fea[:batch // 2]
            im_fea = down_fea[batch // 2:]
            cos_scores = cos_similar(sk_fea, im_fea)  # [2b, 49, 49]
            cos_scores = cos_scores.view(batch // 2, -1)
            rn_scores = self.rn(cos_scores)  # [2b, 1]

            # print('cls_fea:', cls_fea.size())
            # print('rn_scores:', cls_fea.size())
            return cls_fea, rn_scores

        else:
            # 推理阶段
            if only_sa:
                sa_fea, left_tokens, idxs = self.sa(sk)  # self attention ,[b, 197, 768]
                return sa_fea, idxs
            else:
                sk_im = torch.cat((sk, im), dim=0)  #拼接 [2b, 3, 224, 224]
                ca_fea = self.ca(sk_im)  # cross-attention [2b, 197, 768]

                cls_fea = ca_fea[:, 0]  # [2b, 1, 768]， cls的特征
                token_fea = ca_fea[:, 1:]  # [2b, 196, 768]， 所有的token的特征
                batch = token_fea.size(0)  # 2b

                token_fea = token_fea.view(batch, 768, 14, 14)  # [2b, 768, 14, 14]， 所有的token的特征
                down_fea = self.conv2d(token_fea) # [2b, 512, 7, 7]， token特征降维
                down_fea = down_fea.view(batch, 512, 7 * 7)  # [2b, 512, 49]，合并通道
                down_fea = down_fea.transpose(1, 2)  # [2b, 49, 512] # 交换维度

                sk_fea = down_fea[:batch // 2] # [b, 49, 512]，sk的特征
                im_fea = down_fea[batch // 2:]  # [b, 49, 512]，im的特征
                cos_scores = cos_similar(sk_fea, im_fea)  # [b, 49, 49]， 两两计算余弦相似度
                cos_scores = cos_scores.view(batch // 2, -1)  # [b, 49*49]，合并维度， 49*49=2401
                rn_scores = self.rn(cos_scores)  # [b, 1]，计算关系网络的分数

                # print('cls_fea:', cls_fea.size())
                # print('rn_scores:', cls_fea.size())
                return cls_fea, rn_scores

if __name__ == '__main__':
    args = Option().parse()
    sk = torch.rand((4, 224, 224))
    im = torch.rand((4, 224, 224))
    model = Model(args)
    cls_fea, rn_scores = model(sk, im)
