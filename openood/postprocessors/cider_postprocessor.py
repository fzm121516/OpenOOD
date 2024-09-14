from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class CIDERPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(CIDERPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K  # 设定K值，表示最近邻的数量
        self.activation_log = None  # 初始化激活日志
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False  # 设置标志位，用于控制setup方法的执行

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:  # 检查是否已经设置过
            activation_log = []  # 初始化激活日志
            net.eval()  # 将网络设为评估模式
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):  # 遍历训练数据
                    data = batch['data'].cuda()  # 将数据移至GPU

                    feature = net.intermediate_forward(data)  # 获取中间层特征
                    activation_log.append(feature.data.cpu().numpy())  # 将特征移至CPU并添加到激活日志

            self.activation_log = np.concatenate(activation_log, axis=0)  # 将激活日志拼接成一个大的数组
            self.index = faiss.IndexFlatL2(feature.shape[1])  # 创建一个FAISS索引
            self.index.add(self.activation_log)  # 将激活日志添加到索引中
            self.setup_flag = True  # 设置标志位为True，表示已经完成设置
        else:
            pass  # 如果已经设置过，则不再执行

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        feature = net.intermediate_forward(data)  # 获取中间层特征
        D, _ = self.index.search(
            feature.cpu().numpy(),  # 将特征移至CPU并进行搜索
            self.K,  # 使用K值进行最近邻搜索
        )
        kth_dist = -D[:, -1]  # 获取第K个最近邻的距离，并取反
        # 放置虚拟预测结果，因为cider只训练特征提取器
        pred = torch.zeros(len(kth_dist))  # 创建与距离数组相同长度的全零预测张量
        return pred, torch.from_numpy(kth_dist)  # 返回预测结果和第K个最近邻的距离

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]  # 设置超参数K

    def get_hyperparam(self):
        return self.K  # 获取当前的超参数K
