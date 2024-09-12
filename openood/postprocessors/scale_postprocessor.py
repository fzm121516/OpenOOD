from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class ScalePostprocessor(BasePostprocessor):
    def __init__(self, config):
        """
        初始化 ScalePostprocessor 类的实例。

        参数:
        config (Config): 配置对象，包含后处理器的配置参数。
        """
        super(ScalePostprocessor, self).__init__(config)
        # 从配置中获取 postprocessor_args 参数，并提取百分位数（percentile）
        self.args = self.config.postprocessor.postprocessor_args
        self.percentile = self.args.percentile
        # 从配置中获取后处理器的超参数扫描设置
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        执行后处理操作。

        参数:
        net (nn.Module): 神经网络模型。
        data (Any): 输入数据。

        返回:
        tuple: 包含预测类别和能量置信度的元组。
        """
        # 使用模型的 forward_threshold 方法获取输出
        output = net.forward_threshold(data, self.percentile)
        # 获取每个样本的最大值的索引作为预测结果
        _, pred = torch.max(output, dim=1)
        # 计算每个样本的能量置信度，使用 logsumexp 进行计算
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        """
        设置超参数。

        参数:
        hyperparam (list): 包含超参数的列表，这里只用第一个元素作为百分位数。
        """
        self.percentile = hyperparam[0]

    def get_hyperparam(self):
        """
        获取当前的百分位数超参数。

        返回:
        float: 当前设置的百分位数。
        """
        return self.percentile
