import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)          # l1_xyz (B, 3, 512);  l1_points (B, 128, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # l2_xyz (B, 3, 128);  l2_points (B, 256, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # l3_points (B, 1024)
        x = l3_points.view(B, 1024)                      # x (B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))    # x (B, 512)
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))    # x (B, 256)
        x = self.fc3(x)                                  # x (B, num_class)get_model
        x = F.log_softmax(x, -1)                         # x (B, num_class)

        return x, l3_points


class focal_loss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(focal_loss, self).__init__()
        self.alpha = torch.Tensor([alpha, 1 - alpha]).cuda()  # [行人，水马] 权重张量
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target, trans_feat):
        """_summary_

        Args:
            pred (_type_): 预测 [B, C]
            target (_type_): 目标 [B, 1]

        Returns:
            _type_: 损失
        """
        target = target.view(-1, 1)  # 目标真值类别 [B, 1]
        log_prob = pred.gather(1, target).view(-1)  # 对数概率 [B]
        prob = torch.exp(log_prob)  # 普通概率 [B]
        weight = self.alpha.gather(0, target.view(-1))  # 类别权重 [B]
        loss = -weight * ((1 - prob) ** self.gamma) * log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class nll_loss(nn.Module):
    def __init__(self):
        super(nll_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)  # 负对数似然函数
        return total_loss
