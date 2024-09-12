from typing import Any  # 导入 Any 类型，用于表示任意类型的变量
from tqdm import tqdm  # 导入 tqdm，用于显示进度条

import torch  # 导入 PyTorch
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torch.utils.data import DataLoader  # 导入数据加载器

import openood.utils.comm as comm  # 导入 OpenOOD 的通信模块，用于并行处理

class BasePostprocessor:
    def __init__(self, config):
        # 初始化方法，接收配置参数并保存到实例变量
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # 设置方法，接收神经网络模型、ID数据加载器和OOD数据加载器
        # 该方法在子类中可以重写，用于进行特定的设置操作
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # 后处理方法，接收神经网络模型和数据，返回预测结果和置信度
        # @torch.no_grad() 装饰器用于在该方法中禁用梯度计算
        output = net(data)  # 获取模型的输出
        score = torch.softmax(output, dim=1)  # 对输出进行 softmax 处理，得到置信度分数
        conf, pred = torch.max(score, dim=1)  # 获取置信度最高的类别和对应的置信度值
        return pred, conf  # 返回预测结果和置信度

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        # 推理方法，接收神经网络模型、数据加载器和进度显示标志
        pred_list, conf_list, label_list = [], [], []  # 初始化预测结果、置信度和标签的列表
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            # 迭代数据加载器中的批次数据，使用 tqdm 显示进度条
            data = batch['data'].cuda()  # 获取数据并移动到 GPU
            label = batch['label'].cuda()  # 获取标签并移动到 GPU
            pred, conf = self.postprocess(net, data)  # 对数据进行后处理，得到预测结果和置信度

            pred_list.append(pred.cpu())  # 将预测结果移动到 CPU 并添加到列表中
            conf_list.append(conf.cpu())  # 将置信度移动到 CPU 并添加到列表中
            label_list.append(label.cpu())  # 将标签移动到 CPU 并添加到列表中

        # 将列表中的值转换为 numpy 数组
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list  # 返回预测结果、置信度和标签
