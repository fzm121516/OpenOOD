from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

# 归一化函数，将向量规范化为单位长度
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class KNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(KNNPostprocessor, self).__init__(config)  # 调用父类构造函数
        self.args = self.config.postprocessor.postprocessor_args  # 从配置中获取参数
        self.K = self.args.K  # 设置K值
        self.activation_log = None  # 初始化激活日志
        self.args_dict = self.config.postprocessor.postprocessor_sweep  # 获取超参数扫描配置
        self.setup_flag = False  # 初始化设置标志

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:  # 如果尚未设置
            activation_log = []  # 初始化激活日志列表
            net.eval()  # 设置网络为评估模式
            with torch.no_grad():  # 在不计算梯度的上下文中进行操作
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',  # 显示进度条描述
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()  # 将数据移动到GPU
                    data = data.float()  # 转换数据类型为float

                    _, feature = net(data, return_feature=True)  # 获取网络输出和特征
                    activation_log.append(
                        normalizer(feature.data.cpu().numpy()))  # 归一化特征并添加到激活日志中

            self.activation_log = np.concatenate(activation_log, axis=0)  # 将激活日志连接成一个数组
            self.index = faiss.IndexFlatL2(feature.shape[1])  # 创建Faiss索引，使用L2距离
            self.index.add(self.activation_log)  # 将激活日志添加到索引中
            self.setup_flag = True  # 设置完成标志
        else:
            pass  # 如果已设置，什么也不做

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)  # 获取网络输出和特征
        feature_normed = normalizer(feature.data.cpu().numpy())  # 归一化特征
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )  # 在Faiss索引中搜索K个最近邻
        kth_dist = -D[:, -1]  # 获取第K个最近邻的距离（负号是为了排序）
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)  # 获取预测的类别
        return pred, torch.from_numpy(kth_dist)  # 返回预测结果和第K个最近邻的距离

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]  # 设置K值

    def get_hyperparam(self):
        return self.K  # 获取K值
