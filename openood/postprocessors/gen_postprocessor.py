from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor

class GENPostprocessor(BasePostprocessor):
    def __init__(self, config):
        # 初始化GENPostprocessor类
        super().__init__(config)  # 调用基类的初始化方法
        self.args = self.config.postprocessor.postprocessor_args  # 从配置中获取处理器参数
        self.gamma = self.args.gamma  # 从配置中提取gamma参数
        self.M = self.args.M  # 从配置中提取M参数
        self.args_dict = self.config.postprocessor.postprocessor_sweep  # 从配置中获取超参数搜索空间

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # 执行后处理
        output = net(data)  # 将数据传递给网络，得到输出
        score = torch.softmax(output, dim=1)  # 对输出应用softmax函数，计算概率分布
        _, pred = torch.max(score, dim=1)  # 获取最大概率对应的类别作为预测结果
        conf = self.generalized_entropy(score, self.gamma, self.M)  # 计算泛化熵作为置信度
        return pred, conf  # 返回预测结果和置信度

    def set_hyperparam(self, hyperparam: list):
        # 设置超参数
        self.gamma = hyperparam[0]  # 设置gamma参数
        self.M = hyperparam[1]  # 设置M参数

    def get_hyperparam(self):
        # 获取超参数
        return [self.gamma, self.M]  # 返回gamma和M参数

    def generalized_entropy(self, softmax_id_val, gamma=0.1, M=100):
        # 计算泛化熵
        probs = softmax_id_val  # 取softmax后的概率分布
        probs_sorted = torch.sort(probs, dim=1)[0][:, -M:]  # 对概率分布排序并选取前M个
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma),
                           dim=1)  # 计算泛化熵

        return -scores  # 返回负的泛化熵作为置信度
