"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()  # 将模型切换到测试模式：禁用 Dropout, BatchNormalization 等操作

    # 迭代数据集批次
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()  # 加载到GPU

        points = points.transpose(2, 1)     # (B, 3, N)
        pred, _ = classifier(points)        # 前向传播
        pred_choice = pred.data.max(1)[1]   # 预测结果

        # 遍历唯一类别标签
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()  # 计算cat类的TP数量（target == cat 返回真值为cat类的掩码）
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])  # accuray = TP / (TP + FN)
            class_acc[cat, 1] += 1  # 类别出现的批次数

        correct = pred_choice.eq(target.long().data).cpu().sum()        # 当前批次的TP数量
        mean_correct.append(correct.item() / float(points.size()[0]))   # 当前批次的平均准确率

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]     # 所有测试集样本中每个类别的平均准确率
    class_acc = np.mean(class_acc[:, 2])                    # 所有类别的平均准确率
    instance_acc = np.mean(mean_correct)                    # 所有测试集样本的平均准确率

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")  # 创建名为Model的日志记录器
    logger.setLevel(logging.INFO)  # 设置日志记录器的日志类别为INFO，只有INFO以上 (WARNING, ERROR, CRITICAL) 的日志才会被记录
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 日志格式：时间 - 记录器名称 - 级别 - 内容
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))  # 文件处理器：用于将日志信息写入文件
    file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别为 INFO
    file_handler.setFormatter(formatter)  # 日志格式：时间 - 记录器名称 - 级别 - 内容
    logger.addHandler(file_handler)  # 将文件处理器添加到日志记录器上

    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)  # 加载训练集数据到CPU
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)  # 加载测试集数据到CPU

    # 可迭代的数据加载器初始化: 以batch_size 为单位、随机打乱、并行加载训练数据
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    # 可迭代的数据加载器初始化: 按顺序加载测试数据，便于模型评估
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)  # 动态导入模型文件
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))  # 复制模型文件到指定目录
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)  # 创建分类器模型
    criterion = model.get_loss()  # 创建损失函数
    classifier.apply(inplace_relu)  # 将模型中的ReLU激活函数转换为就地操作，使得它直接修改输入的张量，而不返回新的张量

    if not args.use_cpu:
        classifier = classifier.cuda()  # 分类器参数加载到GPU中
        criterion = criterion.cuda()  # 损失函数加载到GPU中

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')  # 加载最优模型文件
        start_epoch = checkpoint['epoch']  # 提取epoch信息
        classifier.load_state_dict(checkpoint['model_state_dict'])  # 加载最优模型的状态字典
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':        # Adam 自适应优化器
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)  # SGD 随机梯度下降优化器

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)  # 学习率调度器

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()  # 将模型切换到训练模式：启用 Dropout, BatchNormalization 等操作
        scheduler.step()  # 检查是否更新学习率

        # 训练：迭代数据集批次
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()  # 优化器梯度清零

            # 数据增强
            points = points.data.numpy()                                                # 将张量转换为NumPy数组 (B, N, 3)
            points = provider.random_point_dropout(points)                              # 随机点丢弃
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])    # 随机尺度缩放
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])           # 随机平移
            points = torch.Tensor(points)    # 将数组转换回张量
            points = points.transpose(2, 1)  # (B, 3, N)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()   # 压入GPU

            pred, trans_feat = classifier(points)               # 前向传播: pred—(B, num_class), trans_feat—(B, 1024)
            loss = criterion(pred, target.long(), trans_feat)   # 损失计算：关联损失和模型
            pred_choice = pred.data.max(1)[1]                   # 预测类别
            loss.backward()                                     # 反向传播
            optimizer.step()                                    # 更新模型参数

            correct = pred_choice.eq(target.long().data).cpu().sum()        # 预测正确的样本数
            mean_correct.append(correct.item() / float(points.size()[0]))   # 计算当前batch的平均正确率
            global_step += 1                                                # 更新全局批次

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        # 测试：不存储中间值进行反向传播
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)  # 测试数据集：实例平均准确率 和 类别平均准确率

            # 检查 实例平均准确率 是否为最优
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            # 检查 类别平均准确率 是否为最优
            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            # 保存最优模型
            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1  # 更新全局epoch

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
