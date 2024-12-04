import os
import torch
import random
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser('SphercialRobotParser')
    parser.add_argument('--data_path', default="/home/luoteng/PointNet2/data/sphercial_robot_data/", type=str, required=False, help='Save results root')
    return parser.parse_args()


def main(args):
    # 写类别文件
    data_root = args.data_path
    cls_list = [item for item in sorted(os.listdir(data_root)) if os.path.isdir(data_root + item)]
    with open(data_root + 'class_names.txt', 'w') as f:
        for cls in cls_list:
            f.write(cls + "\n")

    # 获取所有样本
    dataset = []
    for cls in cls_list:
        sample_list = sorted(os.listdir(data_root + cls))
        dataset.append([cls + "/" + sample for sample in sample_list])

    # 写所有样本文件
    with open(data_root + 'filelist.txt', 'w') as f:
        for cls_data in dataset:
            for sample in cls_data:
                f.write(sample + "\n")

    # 按 8:2 的比例划分训练集和测试集
    train_set = []
    test_set = []
    for cls_data in dataset:
        cls_train, cls_test = train_test_split(cls_data, test_size=0.2, random_state=42)
        train_set.extend(sorted(cls_train))
        test_set.extend(sorted(cls_test))

    with open(data_root + 'robot_train.txt', 'w') as f:
        for sample in train_set:
            f.write(sample + "\n")

    with open(data_root + 'robot_test.txt', 'w') as f:
        for sample in test_set:
            f.write(sample + "\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
