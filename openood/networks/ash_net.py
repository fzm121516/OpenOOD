import numpy as np
import torch
import torch.nn as nn


class ASHNet(nn.Module):
    def __init__(self, backbone):
        super(ASHNet, self).__init__()
        self.backbone = backbone  # 将传入的 backbone 作为特征提取网络

    def forward(self, x, return_feature=False, return_feature_list=False):
        try:
            # 尝试调用 backbone 的前向传播方法，可能返回特征和其他信息
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            # 如果 backbone 不支持 return_feature_list 参数，则只返回特征
            return self.backbone(x, return_feature)

    def forward_threshold(self, x, percentile):
        # 使用 backbone 提取特征，并应用阈值处理
        _, feature = self.backbone(x, return_feature=True)
        feature = ash_b(feature.view(feature.size(0), -1, 1, 1), percentile)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.backbone.get_fc_layer()(feature)  # 通过全连接层进行分类
        return logits_cls

    def get_fc(self):
        # 获取 backbone 中全连接层的权重和偏置
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()


def ash_b(x, percentile=65):
    assert x.dim() == 4  # 确保输入张量是四维的
    assert 0 <= percentile <= 100  # 确保百分位数在合理范围内
    b, c, h, w = x.shape  # 获取张量的形状

    # 计算每个样本的特征总和
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()  # 计算特征的总元素数
    k = n - int(np.round(n * percentile / 100.0))  # 计算阈值所需的元素数量
    t = x.view((b, c * h * w))  # 将张量展平成二维
    v, i = torch.topk(t, k, dim=1)  # 获取前 k 大的元素值和索引
    fill = s1 / k  # 计算每个元素的填充值
    fill = fill.unsqueeze(dim=1).expand(v.shape)  # 扩展填充值的维度
    t.zero_().scatter_(dim=1, index=i, src=fill)  # 将填充值应用到原张量中
    return x


def ash_p(x, percentile=65):
    assert x.dim() == 4  # 确保输入张量是四维的
    assert 0 <= percentile <= 100  # 确保百分位数在合理范围内

    b, c, h, w = x.shape  # 获取张量的形状

    n = x.shape[1:].numel()  # 计算特征的总元素数
    k = n - int(np.round(n * percentile / 100.0))  # 计算阈值所需的元素数量
    t = x.view((b, c * h * w))  # 将张量展平成二维
    v, i = torch.topk(t, k, dim=1)  # 获取前 k 大的元素值和索引
    t.zero_().scatter_(dim=1, index=i, src=v)  # 将前 k 大的值应用到原张量中

    return x


def ash_s(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape
    s1 = x.sum(dim=[1, 2, 3])                       # 计算特征总和
    n = x.shape[1:].numel()                         # 计算特征的总元素数
    k = n - int(np.round(n * percentile / 100.0))   # 计算阈值以上的元素数量k
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)                  # 获取前 k 大的元素值和索引
    t.zero_().scatter_(dim=1, index=i, src=v)       # 将阈值以下的元素置为0
    s2 = x.sum(dim=[1, 2, 3])                       # 计算前 k 大的元素值的总和

    scale = s1 / s2                                 # 计算缩放因子
    x = x * torch.exp(scale[:, None, None, None])   # 对特征进行缩放
    return x


def ash_rand(x, percentile=65, r1=0, r2=10):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape
    n = x.shape[1:].numel()                         # 计算总数
    k = n - int(np.round(n * percentile / 100.0))   # 计算阈值以上的数量
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)  # 获取前 k 大的元素值和索引
    v = v.uniform_(r1, r2)  # 将值随机化到 [r1, r2] 范围内
    t.zero_().scatter_(dim=1, index=i, src=v)  # 将随机化的值应用到原张量中
    return x
