from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.special import logsumexp
from copy import deepcopy
from .base_postprocessor import BasePostprocessor

# 归一化函数，将特征向量归一化到单位范数
normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


# KNN得分计算函数
def knn_score(bankfeas, queryfeas, k=100, min=False):
    # 深拷贝特征数据
    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))

    # 使用Faiss库创建一个基于内积的索引
    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas)

    # 对查询特征进行KNN搜索
    D, _ = index.search(queryfeas, k)

    # 根据min参数选择KNN得分
    if min:
        scores = np.array(D.min(axis=1))  # 取每个查询的最小距离
    else:
        scores = np.array(D.mean(axis=1))  # 取每个查询的平均距离

    return scores


# 基于最近邻的引导后处理类
class NNGuidePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NNGuidePostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K  # 最近邻的K值
        self.alpha = self.args.alpha  # 用于确定训练集特征的比例
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False  # 标记是否已经完成设置

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # 如果尚未完成设置
        if not self.setup_flag:
            net.eval()  # 将网络设置为评估模式
            bank_feas = []
            bank_logits = []
            with torch.no_grad():  # 不计算梯度
                # 遍历训练集数据
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()  # 将数据移到GPU
                    data = data.float()  # 确保数据为浮点型

                    logit, feature = net(data, return_feature=True)  # 获取网络输出的logits和特征
                    bank_feas.append(normalizer(feature.data.cpu().numpy()))  # 归一化特征并保存
                    bank_logits.append(logit.data.cpu().numpy())  # 保存logits

                    # 如果已处理的特征数足够
                    if len(bank_feas
                           ) * id_loader_dict['train'].batch_size > int(
                        len(id_loader_dict['train'].dataset) *
                        self.alpha):
                        break

            # 合并所有特征和logits
            bank_feas = np.concatenate(bank_feas, axis=0)
            bank_confs = logsumexp(np.concatenate(bank_logits, axis=0),
                                   axis=-1)
            self.bank_guide = bank_feas * bank_confs[:, None]  # 计算引导特征

            self.setup_flag = True  # 设置完成
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logit, feature = net(data, return_feature=True)  # 获取网络输出的logits和特征
        feas_norm = normalizer(feature.data.cpu().numpy())  # 归一化特征
        energy = logsumexp(logit.data.cpu().numpy(), axis=-1)  # 计算logits的能量

        conf = knn_score(self.bank_guide, feas_norm, k=self.K)  # 计算KNN得分
        score = conf * energy  # 计算最终得分

        _, pred = torch.max(torch.softmax(logit, dim=1), dim=1)  # 获取预测结果
        return pred, torch.from_numpy(score)  # 返回预测结果和得分

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]  # 设置K值
        self.alpha = hyperparam[1]  # 设置alpha值

    def get_hyperparam(self):
        return [self.K, self.alpha]  # 获取当前的K值和alpha值
