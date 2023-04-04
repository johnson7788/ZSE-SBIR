#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/4/4 15:48
# @File  : generate_data.py
# @Author: 
# @Desc  : 数据集配置生成

import os

def gen_data():
    root_path = "datasets/Sketchy"
    assert os.path.exists(root_path), f"路径{root_path}不存在"
    shot_path = os.path.join(root_path, "zeroshot2")
    if not os.path.exists(shot_path):
        os.mkdir(shot_path)
    image_dir = os.path.join(root_path, "256x256")
    train_photo_file = "all_photo_filelist_train.txt"
    test_photo_file = "all_photo_filelist_zero.txt"
    train_class_file = "cname_cid.txt"
    test_class_file = "cname_cid_zero.txt"
    train_sketch_file = "sketch_tx_000000000000_ready_filelist_train.txt"
    test_sketch_file = "sketch_tx_000000000000_ready_filelist_zero.txt"
    photo_dir = os.path.join(image_dir, "photo/tx_000000000000")
    sketch_dir = os.path.join(image_dir, "sketch/tx_000000000000")
    train_class_names = os.listdir(photo_dir)
    test_class_names = os.listdir(sketch_dir)
    assert train_class_names == test_class_names, "训练集和测试集的类别不一致"
    print(f"训练集的类别数量为{len(train_class_names)}")
    # 前100类作为训练集，剩下的类作为测试集
    train_class_names = train_class_names[:100]
    test_class_names = test_class_names[100:]
    train_class_dict = {class_name: i for i, class_name in enumerate(train_class_names)}
    test_class_dict = {class_name: i for i, class_name in enumerate(test_class_names)}
    # 保存训练集的类别
    with open(os.path.join(shot_path, train_class_file), "w") as f:
        for class_name in train_class_names:
            f.write(f"{class_name} {train_class_dict[class_name]}\n")
    # 保存测试集的类别
    with open(os.path.join(shot_path, test_class_file), "w") as f:
        for class_name in test_class_names:
            f.write(f"{class_name} {test_class_dict[class_name]}\n")
    # 保存photo训练集的图片
    train_photo_data = []
    for class_name in train_class_names:
        class_dir = os.path.join(photo_dir, class_name)
        # 去掉root_path
        sub_dir = class_dir[len(root_path) + 1:]
        for file_name in os.listdir(class_dir):
            train_photo_data.append(f"{sub_dir}/{file_name} {train_class_dict[class_name]}\n")
    with open(os.path.join(shot_path, train_photo_file), "w") as f:
        f.writelines(train_photo_data)
    # 保存photo测试集的图片
    test_photo_data = []
    for class_name in test_class_names:
        class_dir = os.path.join(photo_dir, class_name)
        sub_dir = class_dir[len(root_path) + 1:]
        for file_name in os.listdir(class_dir):
            test_photo_data.append(f"{sub_dir}/{file_name} {test_class_dict[class_name]}\n")
    with open(os.path.join(shot_path, test_photo_file), "w") as f:
        f.writelines(test_photo_data)
    # 保存sketch训练集的图片
    train_sketch_data = []
    for class_name in train_class_names:
        class_dir = os.path.join(sketch_dir, class_name)
        sub_dir = class_dir[len(root_path) + 1:]
        for file_name in os.listdir(class_dir):
            train_sketch_data.append(f"{sub_dir}/{file_name} {train_class_dict[class_name]}\n")
    with open(os.path.join(shot_path, train_sketch_file), "w") as f:
        f.writelines(train_sketch_data)
    # 保存sketch测试集的图片
    test_sketch_data = []
    for class_name in test_class_names:
        class_dir = os.path.join(sketch_dir, class_name)
        sub_dir = class_dir[len(root_path) + 1:]
        for file_name in os.listdir(class_dir):
            test_sketch_data.append(f"{sub_dir}/{file_name} {test_class_dict[class_name]}\n")
    with open(os.path.join(shot_path, test_sketch_file), "w") as f:
        f.writelines(test_sketch_data)
    print(f"保存数据集配置成功，保存路径为{shot_path}")

if __name__ == '__main__':
    gen_data()