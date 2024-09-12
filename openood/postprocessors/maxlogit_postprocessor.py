from typing import Any  # 导入 Any 类型，用于类型注释

import numpy as np  # 导入 numpy 库，通常用于数值计算
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条

from .base_postprocessor import BasePostprocessor  # 从当前模块导入 BasePostprocessor 类


class MaxLogitPostprocessor(BasePostprocessor):  # 定义 MaxLogitPostprocessor 类，继承自 BasePostprocessor
    def __init__(self, config):  # 构造函数，接受配置参数
        super().__init__(config)  # 调用基类的构造函数
        self.args = self.config.postprocessor.postprocessor_args  # 从配置中提取 postprocessor_args

    @torch.no_grad()  # 指示在此方法中不计算梯度，以节省内存和计算资源
    def postprocess(self, net: nn.Module, data: Any):  # 定义 postprocess 方法，接受神经网络模型和数据
        output = net(data)  # 使用神经网络模型对输入数据进行前向传播，获取输出
        conf, pred = torch.max(output, dim=1)  # 在输出的每一行中找到最大值及其索引
        return pred, conf  # 返回预测标签和对应的置信度
