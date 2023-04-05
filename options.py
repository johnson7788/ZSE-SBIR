import argparse


class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")

        # dataset
        parser.add_argument('--data_path', type=str, default="./datasets")
        parser.add_argument('--dataset', type=str, default='comestic',
                            choices=['sketchy_extend', 'tu_berlin', 'Quickdraw','comestic'])
        parser.add_argument('--test_class', type=str, default='test_class_sketchy2',
                            choices=['test_class_sketchy2','test_class_sketchy25', 'test_class_sketchy21', 'test_class_tuberlin30', 'Quickdraw','test_class_comestic'])

        # model
        parser.add_argument('--cls_number', type=int, default=100)
        parser.add_argument('--d_model', type=int, default=768)
        parser.add_argument('--d_ff', type=int, default=1024)
        parser.add_argument('--head', type=int, default=8)
        parser.add_argument('--number', type=int, default=1)
        parser.add_argument('--pretrained', default=True, action='store_false')
        parser.add_argument('--anchor_number', '-a', type=int, default=49)

        # train
        parser.add_argument('--save', '-s', type=str, default='./checkpoints/sketchy_ext')
        parser.add_argument('--batch', type=int, default=15)
        parser.add_argument('--epoch', type=int, default=30)
        parser.add_argument('--datasetLen', type=int, default=10000)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=1e-2)

        # test
        parser.add_argument('--load', '-l', type=str, default=None)
        parser.add_argument('--retrieval', '-r', type=str, default='rn', choices=['rn', 'sa'])
        parser.add_argument('--testall', default=False, action='store_true', help='train/test scale')
        parser.add_argument('--test_sk', type=int, default=20)
        parser.add_argument('--test_im', type=int, default=20)
        parser.add_argument('--num_workers', type=int, default=0)

        # other
        # 增加只用CPU的选项
        parser.add_argument('--cpu',action='store_true', help='只用CPU')
        parser.add_argument('--choose_cuda', '-c', type=str, default='0')
        parser.add_argument("--seed", type=int, default=2021, help="random seed.")

        self.parser = parser


    def parse(self):
        return self.parser.parse_args()
