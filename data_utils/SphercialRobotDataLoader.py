'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    if N <= npoint:
        return point
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))  # 创建长度为npoints的元组
    distance = np.ones((N,)) * 1e10  # 距离上限
    farthest = np.random.randint(0, N)  # 随机选取质心点
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)  # 计算所有点到质心点的距离
        mask = dist < distance  # 对应点的距离是否小于当前记录的最小距离
        distance[mask] = dist[mask]  # 更新最小距离
        farthest = np.argmax(distance, -1)  # 迭代作为质心点
    point = point[centroids.astype(np.int32)]
    return point


class RobotDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.npoints = args.num_point  # 每个样本采样点数
        self.use_normals = args.use_normals  # 是否使用法向量特征
        self.num_category = args.num_category  # 类别数
        self.catfile = os.path.join(self.root, 'class_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  # 类别名称映射到一个唯一整数ID
        self.class_ids = dict(zip(range(len(self.cat)), self.cat))  # ID映射到类别名称

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'robot_train.txt'))]  # 划分训练集
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'robot_test.txt'))]  # 划分测试集

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('/')[0:-1]) for x in shape_ids[split]]  # 提取数据集类别标签
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_ids[split][i])) for i in range(len(shape_ids[split]))]  # [(类别名, 点云文件路径)]
        print('The size of %s data is %d' % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        label = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter=' ').astype(np.float32)
        point_set = farthest_point_sample(point_set, self.npoints)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if (point_set.shape[0] < self.npoints):
            padding = np.zeros((self.npoints - point_set.shape[0], 3 + self.use_normals * 3))
            point_set = np.vstack((point_set, padding))

        return point_set, label[0]


if __name__ == '__main__':
    import torch

    data = RobotDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
