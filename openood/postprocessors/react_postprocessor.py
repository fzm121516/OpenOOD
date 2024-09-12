from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class ReactPostprocessor(BasePostprocessor):
    def __init__(self, config):
        # 调用父类构造函数
        super(ReactPostprocessor, self).__init__(config)
        # 获取配置中的参数
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile  # 百分位数
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False  # 设置标志，表示是否已完成初始化设置

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # 如果尚未完成初始化设置
        if not self.setup_flag:
            activation_log = []  # 用于存储激活日志
            net.eval()  # 将网络设置为评估模式
            with torch.no_grad():  # 禁用梯度计算
                for batch in tqdm(id_loader_dict['val'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    # 获取数据并将其转移到GPU
                    data = batch['data'].cuda()
                    data = data.float()  # 将数据转换为浮点型

                    # 前向传播，获取特征
                    _, feature = net(data, return_feature=True)
                    # 将特征数据从GPU转移到CPU，并保存到激活日志中
                    activation_log.append(feature.data.cpu().numpy())

            # 将激活日志转换为NumPy数组
            self.activation_log = np.concatenate(activation_log, axis=0)
            self.setup_flag = True  # 设置标志，表示初始化设置已完成
        else:
            pass

        # 根据百分位数计算阈值
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # 使用阈值进行前向传播
        output = net.forward_threshold(data, self.threshold)
        # 计算softmax分数
        score = torch.softmax(output, dim=1)
        # 获取预测结果
        _, pred = torch.max(score, dim=1)
        # 计算能量置信度
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        # 设置超参数
        self.percentile = hyperparam[0]
        # 重新计算阈值
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))

    def get_hyperparam(self):
        # 获取当前的百分位数
        return self.percentile
