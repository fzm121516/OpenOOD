import numpy as np
import torch
import torch.nn as nn


class ScaleNet(nn.Module):
    def __init__(self, backbone):
        """
        初始化 ScaleNet 类的实例。

        参数:
        backbone (nn.Module): 用作特征提取的基础网络模型。
        """
        super(ScaleNet, self).__init__()
        # 将 backbone 赋值为模型的成员变量
        self.backbone = backbone

    def forward(self, x, return_feature=False, return_feature_list=False):
        """
        执行前向传播操作。

        参数:
        x (Tensor): 输入数据。
        return_feature (bool): 是否返回特征（默认为 False）。
        return_feature_list (bool): 是否返回特征列表（默认为 False）。

        返回:
        Tensor 或 tuple: 如果 return_feature 或 return_feature_list 为 True，返回 tuple；否则返回 Tensor。
        """
        try:
            # 尝试调用 backbone 的前向传播方法
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            # 如果遇到 TypeError，则调用不包含 return_feature_list 的方法
            return self.backbone(x, return_feature)

    def forward_threshold(self, x, percentile):
        """
        使用指定的百分位数进行特征缩放，并返回分类 logits。

        参数:
        x (Tensor): 输入数据。
        percentile (float): 百分位数，用于特征缩放。

        返回:
        Tensor: 分类 logits。
        """
        # 获取 backbone 提取的特征
        _, feature = self.backbone(x, return_feature=True)
        # 使用 scale 函数对特征进行缩放
        feature = scale(feature.view(feature.size(0), -1, 1, 1), percentile)
        # 将特征展平为 (batch_size, num_features)
        feature = feature.view(feature.size(0), -1)
        # 通过全连接层获取 logits
        logits_cls = self.backbone.get_fc_layer()(feature)
        return logits_cls

    def get_fc(self):
        """
        获取全连接层的权重和偏置。

        返回:
        tuple: 包含权重和偏置的元组，它们被转换为 NumPy 数组。
        """
        # 获取 backbone 中的全连接层
        fc = self.backbone.fc
        # 将权重和偏置从 GPU 转移到 CPU，并转为 NumPy 数组
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()


def scale(x, percentile=65):
    input = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape
    s1 = x.sum(dim=[1, 2, 3])                       # 计算特征总和
    n = x.shape[1:].numel()                         # 计算特征的总元素数
    k = n - int(np.round(n * percentile / 100.0))   # 计算阈值以上的元素数量k
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)                  # 获取前 k 大的元素值和索引
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=[1, 2, 3])                       # 计算前 k 大的元素值的总和

    scale = s1 / s2                                 # 计算缩放因子

    return input * torch.exp(scale[:, None, None, None])  # 输入乘以缩放因子
