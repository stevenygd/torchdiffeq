from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F


#############################
# AtlasNet PointNet Encoder #
#############################
class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class AtlasNetPointNet(nn.Module):

    def __init__(self, cdim):
        super(AtlasNetPointNet, self).__init__()
        self.bottleneck_size = cdim
        self.pointnet_feat = PointNetfeat()
        self.fc = nn.Linear(1024, self.bottleneck_size)
        self.bn = nn.BatchNorm1d(self.bottleneck_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.pointnet_feat(x)
        # x = self.relu(self.bn(self.fc(x)))
        x = self.bn(self.fc(x))
        return x


class AtlasNetPointNetStochastic(nn.Module):

    def __init__(self, cdim):
        super(AtlasNetPointNetStochastic, self).__init__()
        self.bottleneck_size = cdim
        self.pointnet_feat = PointNetfeat()
        self.fc_mean = nn.Linear(1024, self.bottleneck_size)
        self.bn_mean = nn.BatchNorm1d(self.bottleneck_size)

        self.fc_var = nn.Linear(1024, self.bottleneck_size)
        self.bn_var = nn.BatchNorm1d(self.bottleneck_size)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.pointnet_feat(x)
        m = self.bn_mean(self.fc_mean(x))
        v = self.bn_var(self.fc_var(x))
        return m, v


############################
# For Nerual statistitcian #
############################
# Statistician Network for 3D point clouds
class PointNetStats(nn.Module):
    def __init__(self, cdim, pool_type='max'):
        super(PointNetStats, self).__init__()
        self.pool_type = 'max'
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, cdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, cdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)


    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64,   #points)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128,  #points)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128,  #points)
        x = self.bn4(self.conv4(x))          # (B, 1024, #points)
        if self.pool_type == 'max':
            x = torch.max(x, 2, keepdim=True)[0] # (B, 1024, 1) -> max pooling
        elif self.pool_type == 'mean':
            x = torch.mean(x, 2, keepdim=True) # (B, 1024, 1) -> mean pooling
        else:
            raise Exception("Invalid pool type:%s"%selfpool_type)

        x = x.view(-1, 512)                 # (B, 1024) -> global feature

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)

        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)
        return m, v


# Encoder from Achlioptas' paper
class L3DPPointNetEncoder(nn.Module):
    def __init__(self, cdim, pool_type='max'):
        super(L3DPPointNetEncoder, self).__init__()
        self.pool_type = 'max'
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c]
        self.fc1 = torch.nn.Conv1d(512, 256, 1)
        self.fc2 = torch.nn.Conv1d(256, 128, 1)
        self.fc3 = torch.nn.Conv1d(128, cdim, 1)
        # self.fc1 = nn.Linear(512, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, cdim)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc_bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64,   #points)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128,  #points)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 128,  #points)
        x = self.bn4(self.conv4(x))          # (B, 1024, #points)
        if self.pool_type == 'max':
            x = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1) -> max pooling
        elif self.pool_type == 'mean':
            x = torch.mean(x, 2, keepdim=True)  # (B, 1024, 1) -> mean pooling
        else:
            raise Exception("Invalid pool type:%s" % self.pool_type)

        # x = x.view(-1, 512)                 # (B, 1024) -> global feature
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.squeeze()
        return x


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetStats(cdim = 128)
    out, _ = cls(sim_data)
    print('class', out.size())

