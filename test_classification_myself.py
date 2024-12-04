"""
Author: Benny
Date: Nov 2019
"""
from data_utils.SphercialRobotDataLoader import RobotDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import datetime
import provider
import open3d as o3d
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

pc_index = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--num_category', default=2, type=int, help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=128, help='Point Number')
    # parser.add_argument('--log_dir', default="pointnet2_cls_ssg", type=str, required=False, help='Experiment root')
    parser.add_argument('--log_dir', default="myselt_focal_loss", type=str, required=False, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def vis_confusion_matrix(mat, cats):
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", xticklabels=cats, yticklabels=cats)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def save(dir, pcs, labels=None, class_ids=None):
    """
    保存点云数据到指定目录

    参数:
    - dir: str, 保存点云数据的目录路径
    - pcs: Tensor, 形状为 [B,N,3] 的点云数据
    - labels: Tensor, 形状为 [B] 的点云标签
    - class_ids: dict, 点云标签到类别名的映射
    """
    global pc_index

    if labels is not None and class_ids is not None:
        save_flag = True
        cats = labels.cpu().numpy()
    else:
        save_flag = False

    for batch_idx in range(pcs.size(0)):
        points = pcs[batch_idx].cpu().numpy()[:, : 3]  # 转换为NumPy数组
        pcd = o3d.geometry.PointCloud()                 # 创建open3d点云对象
        pcd.points = o3d.utility.Vector3dVector(points)
        file_path = os.path.join(dir, ("%d%s.pcd" % (pc_index, class_ids[cats[batch_idx]]))) if save_flag else os.path.join(dir, ("%d.pcd" % pc_index))      # 保存文件路径
        o3d.io.write_point_cloud(file_path, pcd)        # 保存点云为pcd格式
        pc_index += 1


def test(model, loader, class_ids, save_dir, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()  # 切换到测试模式：禁用 Dropout 和 BatchNormalization 等
    class_acc = np.zeros((num_class, 3))

    confusion_matrix = np.zeros((class_ids.__len__(), class_ids.__len__()), dtype=int)  # 初始化混淆矩阵

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points = points.data.numpy()
        points = provider.occlusion_point_dropout(points, 0.3, 0.9)                              # 随机点丢弃
        points = torch.Tensor(points)

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()  # 压入GPU

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        # 多次预测取平均得分
        for _ in range(vote_num):
            pred, _ = classifier(points)   # 前向传播
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]  # 预测结果
        points = points.transpose(1, 2)

        # 统计混淆矩阵
        for t, p in zip(target.cpu().numpy(), pred_choice.cpu().numpy()):
            confusion_matrix[t, p] += 1

        # 遍历批次点云
        for cat in np.unique(target.cpu()):
            cat_pred = pred_choice[target == cat]                           # 提取当前类别的预测值
            cat_points = points[target == cat]                              # 提取当前类别的点云
            tp_mask = cat_pred.eq(cat)                                      # TP掩码
            tp_dir = save_dir.joinpath(class_ids[cat] + '/TP')      # TP保存目录
            fn_dir = save_dir.joinpath(class_ids[cat] + '/FN')      # FN保存目录
            save(tp_dir, cat_points[tp_mask])
            save(fn_dir, cat_points[~tp_mask], cat_pred[~tp_mask], class_ids)

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc, confusion_matrix


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")                                                     # 创建名为Model的日志记录器
    logger.setLevel(logging.INFO)                                                           # 设置日志记录器的日志类别为INFO
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')   # 日志格式：时间 - 记录器名称 - 级别 - 内容
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)                      # 创建文件处理器：用于将日志信息写入文件
    file_handler.setLevel(logging.INFO)                                                     # 设置文件处理器的日志级别为 INFO
    file_handler.setFormatter(formatter)                                                    # 日志格式：时间 - 记录器名称 - 级别 - 内容
    logger.addHandler(file_handler)                                                         # 将文件处理器添加到日志记录器上
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/sphercial_robot_data/'
    test_dataset = RobotDataLoader(root=data_path, args=args, split='test')                          # 加载测试集数据
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)   # 创建数据迭代器

    '''创建测试结果目录'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    test_dir = Path(experiment_dir + '/test_results/')
    test_dir.mkdir(exist_ok=True)
    test_dir = test_dir.joinpath(timestr)
    test_dir.mkdir(exist_ok=True)
    catfile = os.path.join(data_path, 'class_names.txt')
    for line in open(catfile):
        line = line.rstrip()
        cat_dir = test_dir.joinpath(line)
        cat_dir.mkdir(exist_ok=True)
        cat_dir.joinpath('TP').mkdir(exist_ok=True)
        cat_dir.joinpath('FN').mkdir(exist_ok=True)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)  # 动态加载模型

    classifier = model.get_model(num_class, normal_channel=args.use_normals)  # 创建分类器模型
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')    # 加载最优模型文件
    classifier.load_state_dict(checkpoint['model_state_dict'])                      # 加载最优模型的参数

    with torch.no_grad():  # 不存储中间值进行反向传播
        instance_acc, class_acc, confusion_matrix = test(classifier.eval(), testDataLoader, test_dataset.class_ids, test_dir, vote_num=args.num_votes, num_class=num_class)  # 测试数据集
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

        # 可视化混淆矩阵
        print("Confusion Matrix:", confusion_matrix)
        vis_confusion_matrix(confusion_matrix, test_dataset.cat)


if __name__ == '__main__':
    args = parse_args()
    main(args)
