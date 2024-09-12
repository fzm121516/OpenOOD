from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor

class ASHPostprocessor(BasePostprocessor):
    def __init__(self, config):
        # 初始化ASHPostprocessor类
        super(ASHPostprocessor, self).__init__(config)  # 调用基类的初始化方法
        self.args = self.config.postprocessor.postprocessor_args  # 从配置中获取处理器参数
        self.percentile = self.args.percentile  # 从配置中提取百分位数参数
        self.args_dict = self.config.postprocessor.postprocessor_sweep  # 从配置中获取超参数搜索空间

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # 执行后处理
        output = net.forward_threshold(data, self.percentile)  # 使用网络的forward_threshold方法进行前向计算，并根据百分位数进行阈值处理
        _, pred = torch.max(output, dim=1)  # 获取最大输出值对应的类别作为预测结果
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)  # 计算对数和指数（Log-Sum-Exp）作为能量置信度
        return pred, energyconf  # 返回预测结果和能量置信度

    def set_hyperparam(self, hyperparam: list):
        # 设置超参数
        self.percentile = hyperparam[0]  # 设置百分位数参数

    def get_hyperparam(self):
        # 获取超参数
        return self.percentile  # 返回百分位数参数
