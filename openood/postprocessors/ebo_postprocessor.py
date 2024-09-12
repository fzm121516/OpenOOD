from typing import Any  # 导入 Any 类型，用于函数参数类型的注解

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 从 PyTorch 中导入神经网络模块

from .base_postprocessor import BasePostprocessor  # 从 base_postprocessor 模块中导入 BasePostprocessor 类


class EBOPostprocessor(BasePostprocessor):  # 定义 EBOPostprocessor 类，继承自 BasePostprocessor
    def __init__(self, config):  # 构造函数，接受配置参数 config
        super().__init__(config)  # 调用父类的构造函数
        self.args = self.config.postprocessor.postprocessor_args  # 从配置中获取 postprocessor_args
        self.temperature = self.args.temperature  # 获取温度参数
        self.args_dict = self.config.postprocessor.postprocessor_sweep  # 获取 postprocessor_sweep 配置

    @torch.no_grad()  # 表示在这个函数中不计算梯度
    def postprocess(self, net: nn.Module, data: Any):  # 定义后处理函数，接受网络模型 net 和数据 data
        output = net(data)  # 将数据传入网络，获取输出
        score = torch.softmax(output, dim=1)  # 对网络输出进行 softmax 操作，计算每个类别的概率分布
        _, pred = torch.max(score, dim=1)  # 获取概率分布中最大值的索引作为预测类别
        conf = self.temperature * torch.logsumexp(output / self.temperature,  # 计算置信度，通过温度缩放和 logsumexp 函数
                                                  dim=1)
        return pred, conf  # 返回预测结果和置信度

    def set_hyperparam(self, hyperparam: list):  # 设置超参数，接受一个超参数列表
        self.temperature = hyperparam[0]  # 将列表中的第一个元素作为温度参数

    def get_hyperparam(self):  # 获取当前超参数
        return self.temperature  # 返回当前的温度参数
