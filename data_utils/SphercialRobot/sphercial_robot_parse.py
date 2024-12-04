import os
import json
import torch
import argparse
import open3d as o3d
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

cls_list = {"Pedestrian", "RoadBarrel"}


class RobotDataLoader(Dataset):
    def __init__(self, root, bag):
        super().__init__()
        self.root = root
        self.bag = bag
        self.bag_path = os.path.join(self.root, self.bag)
        self.jsons_path = os.path.join(self.bag_path, "label")
        self.pcds_path = os.path.join(self.bag_path, "lidar")

        # 点云和json文件目录检验
        pcd_list = sorted(os.listdir(self.pcds_path))
        json_list = sorted(os.listdir(self.jsons_path))
        if (len(pcd_list) != len(json_list)):
            for i in range(len(pcd_list)):
                pcd = pcd_list[i].rsplit('.', maxsplit=1)[0]
                jsn = json_list[i].rsplit('.', maxsplit=1)[0]
                assert pcd == jsn, ("%s: PCD files and JSON files do not match." % bag)

        # 加载点云和json文件目录
        self.dataset = [(os.path.join(self.pcds_path, pcd_list[i]), os.path.join(self.jsons_path, json_list[i])) for i in range(len(pcd_list))]
        print('The size of %s dataset is %d' % (self.bag, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pcd_file = self.dataset[index][0]
        json_file = self.dataset[index][1]
        pcd = o3d.io.read_point_cloud(pcd_file)
        pnts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
        objs = self.load_json(json_file)
        return pnts, objs

    def load_json(self, json_file):
        """ 加载json文件

        Args:
            json_file (_type_): json文件
        """
        with open(json_file, 'r') as f:
            content = f.read().strip()
        json_msg = json.loads(content)

        objs = []
        for obj in json_msg:
            label = obj["obj_type"]
            position = obj["psr"]["position"]
            scale = obj["psr"]["scale"]
            objs.append([label,
                         torch.tensor([position["x"], position["y"], position["z"]], dtype=torch.float32),
                         torch.tensor([scale["x"], scale["y"], scale["z"]], dtype=torch.float32)])
        return objs


def parse_args():
    parser = argparse.ArgumentParser('SphercialRobotParser')
    parser.add_argument('--data_path', default="/home/luoteng/dataset/SphercialRobotDataset/", type=str, required=False, help='Experimental dataset root')
    parser.add_argument('--save_path', default="/home/luoteng/PointNet2/data/sphercial_robot_data/", type=str, required=False, help='Save results root')
    parser.add_argument('--save_txt_format', default=True, type=bool, required=False, help='Save results format.')
    parser.add_argument('--save_normals', default=True, type=bool, required=False, help='Save normal infomation')
    return parser.parse_args()


def visual_pcds(src_pc, dst_pc, label):
    """可视化点云

    Args:
        src (_type_): _description_ 源点云
        dst (_type_): _description_ 目标点云
    """
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pc)
    src.paint_uniform_color([0, 1, 0])
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(dst_pc)
    dst.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([src, dst], label)

# 计算每个点的法向量


def save_pc_pcd(pc_data, file_path, save_normals=False):
    """将点云保存为pcd文件

    Args:
        pc_data (_type_): 点云文件
        file_path (_type_): 保存路径
        save_normals (_type_): 保存法向量

    Returns:
        _type_: True 保存成功; False 保存失败
    """
    if pc_data.shape[0] < 30:
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data)
    if save_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))

    o3d.io.write_point_cloud(str(file_path), pcd, write_ascii=False)


def save_pc_txt(pc_data, file_path, save_normals=False):
    """将点云保存为txt文件

    Args:
        pc_data (_type_): 点云文件
        file_path (_type_): 保存路径
        save_normals (_type_): 保存法向量

    Returns:
        _type_: True 保存成功; False 保存失败
    """
    if pc_data.shape[0] < 30:
        return False

    if save_normals:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_data)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
        pc_data = np.hstack((pc_data, np.asarray(pcd.normals)))

    np.savetxt(str(file_path), pc_data, fmt="%.6f")
    return True


def main(args):
    # 创建相关目录
    save_root = args.save_path
    cls_index = {}
    for cls in cls_list:
        cls_path = Path(save_root, cls)
        cls_path.mkdir(parents=True, exist_ok=True)
        cls_index[cls] = 0

    # 读取原始数据包
    data_root = args.data_path
    bags = os.listdir(data_root)
    for bag in bags:
        dataset = RobotDataLoader(data_root, bag)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)
        for j, (pc, objs) in tqdm(enumerate(data_loader), total=len(data_loader)):
            pc = pc.reshape(-1, 3)
            for obj in objs:
                label = obj[0][0]
                center = obj[1][0]
                scale = obj[2][0]
                max_pt = center + scale / 2
                min_pt = center - scale / 2
                mask = torch.all((min_pt < pc) & (pc < max_pt), dim=1)
                obj_pc = pc[mask]

                if args.save_txt_format:
                    save_file = Path(save_root, label).joinpath(label + "_%05d.txt" % cls_index[label])
                    if save_pc_txt(obj_pc, save_file, args.save_normals):
                        cls_index[label] += 1
                else:
                    save_file = Path(save_root, label).joinpath(label + "_%05d.pcd" % cls_index[label])
                    if save_pc_pcd(obj_pc, save_file, args.save_normals):
                        cls_index[label] += 1

                # visual_pcds(pc, obj_pc, label)  # 可视化

    return


if __name__ == '__main__':
    args = parse_args()
    main(args)
