from typing import Callable, List, Type

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor
from openood.networks.ash_net import ASHNet
from openood.networks.react_net import ReactNet
from openood.networks.scale_net import ScaleNet

from .datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor


class Evaluator:
    def __init__(
            self,
            net: nn.Module,
            id_name: str,
            data_root: str = './data',
            config_root: str = './configs',
            preprocessor: Callable = None,
            postprocessor_name: str = None,
            postprocessor: Type[BasePostprocessor] = None,
            batch_size: int = 200,
            shuffle: bool = False,
            num_workers: int = 4,
    ) -> None:
        """用于评估大多数判别式 OOD 检测方法的统一易用 API。

        Args:
            net (nn.Module):
                基础分类器。
            id_name (str):
                归属数据集的名称。
            data_root (str, optional):
                数据文件夹的路径。默认为 './data'。
            config_root (str, optional):
                配置文件夹的路径。默认为 './configs'。
            preprocessor (Callable, optional):
                输入图像的预处理器。
                如果传入 None，将使用默认预处理器。
                默认为 None。
            postprocessor_name (str, optional):
                用于获取 OOD 分数的后处理器名称。
                如果传入实际的后处理器则忽略。
                默认为 None。
            postprocessor (Type[BasePostprocessor], optional):
                继承自 OpenOOD 的 BasePostprocessor 的实际后处理器实例。默认为 None。
            batch_size (int, optional):
                样本的批量大小。默认为 200。
            shuffle (bool, optional):
                是否打乱样本。默认为 False。
            num_workers (int, optional):
                传递给数据加载器的 num_workers 参数。默认为 4。

        Raises:
            ValueError:
                如果 postprocessor_name 和 postprocessor 都为 None。
            ValueError:
                如果指定的 ID 数据集 {id_name} 不受支持。
            TypeError:
                如果传递的后处理器不继承自 BasePostprocessor。
        """
        # 检查参数
        if postprocessor_name is None and postprocessor is None:
            raise ValueError('请传入 postprocessor_name 或 postprocessor')
        if postprocessor_name is not None and postprocessor is not None:
            print(
                '因为传入了 postprocessor，所以 postprocessor_name 被忽略'
            )
        if id_name not in DATA_INFO:
            raise ValueError(f'数据集 [{id_name}] 不受支持')

        # 获取数据预处理器
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        # 设置配置文件夹路径
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split('/')[:-2], 'configs')

        # 获取后处理器
        if postprocessor is None:
            postprocessor = get_postprocessor(config_root, postprocessor_name,
                                              id_name)
        if not isinstance(postprocessor, BasePostprocessor):
            raise TypeError(
                'postprocessor 应该继承自 OpenOOD 的 BasePostprocessor')

        # 加载数据
        data_setup(data_root, id_name)
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                                preprocessor, **loader_kwargs)

        # 包装基础模型以适应特定的后处理器
        if postprocessor_name == 'react':
            net = ReactNet(net)
        elif postprocessor_name == 'ash':
            net = ASHNet(net)
        elif postprocessor_name == 'scale':
            net = ScaleNet(net)

        # 后处理器设置
        postprocessor.setup(net, dataloader_dict['id'], dataloader_dict['ood'])

        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict
        self.metrics = {
            'id_acc': None,
            'csid_acc': None,
            'ood': None,
            'fsood': None
        }
        self.scores = {
            'id': {
                'train': None,
                'val': None,
                'test': None
            },
            'csid': {k: None
                     for k in dataloader_dict['csid'].keys()},
            'ood': {
                'val': None,
                'near':
                    {k: None
                     for k in dataloader_dict['ood']['near'].keys()},
                'far': {k: None
                        for k in dataloader_dict['ood']['far'].keys()},
            },
            'id_preds': None,
            'id_labels': None,
            'csid_preds': {k: None
                           for k in dataloader_dict['csid'].keys()},
            'csid_labels': {k: None
                            for k in dataloader_dict['csid'].keys()},
        }
        # 如果还未进行超参数搜索，则执行
        if (self.postprocessor.APS_mode
                and not self.postprocessor.hyperparam_search_done):
            self.hyperparam_search()

        self.net.eval()

    def _classifier_inference(self,
                              data_loader: DataLoader,
                              msg: str = 'Acc Eval',
                              progress: bool = True):
        """对数据加载器进行分类器推断。

        Args:
            data_loader (DataLoader):
                用于推断的数据加载器。
            msg (str, optional):
                进度条的描述信息。默认为 'Acc Eval'。
            progress (bool, optional):
                是否显示进度条。默认为 True。

        Returns:
            all_preds (torch.Tensor):
                所有预测结果。
            all_labels (torch.Tensor):
                所有标签。
        """
        self.net.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress):
                data = batch['data'].cuda()
                logits = self.net(data)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(batch['label'])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels

    def eval_acc(self, data_name: str = 'id') -> float:
        """评估准确率。

        Args:
            data_name (str, optional):
                数据名称（'id' 或 'csid'）。默认为 'id'。

        Returns:
            acc (float):
                准确率。
        """
        if data_name == 'id':
            if self.metrics['id_acc'] is not None:
                return self.metrics['id_acc']
            else:
                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                assert len(all_preds) == len(all_labels)
                correct = (all_preds == all_labels).sum().item()
                acc = correct / len(all_labels) * 100
                self.metrics['id_acc'] = acc
                return acc
        elif data_name == 'csid':
            if self.metrics['csid_acc'] is not None:
                return self.metrics['csid_acc']
            else:
                correct, total = 0, 0
                for _, (dataname, dataloader) in enumerate(
                        self.dataloader_dict['csid'].items()):
                    if self.scores['csid_preds'][dataname] is None:
                        all_preds, all_labels = self._classifier_inference(
                            dataloader, f'CSID {dataname} Acc Eval')
                        self.scores['csid_preds'][dataname] = all_preds
                        self.scores['csid_labels'][dataname] = all_labels
                    else:
                        all_preds = self.scores['csid_preds'][dataname]
                        all_labels = self.scores['csid_labels'][dataname]

                    assert len(all_preds) == len(all_labels)
                    c = (all_preds == all_labels).sum().item()
                    t = len(all_labels)
                    correct += c
                    total += t

                if self.scores['id_preds'] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict['id']['test'], 'ID Acc Eval')
                    self.scores['id_preds'] = all_preds
                    self.scores['id_labels'] = all_labels
                else:
                    all_preds = self.scores['id_preds']
                    all_labels = self.scores['id_labels']

                correct += (all_preds == all_labels).sum().item()
                total += len(all_labels)

                acc = correct / total * 100
                self.metrics['csid_acc'] = acc
                return acc
        else:
            raise ValueError(f'未知的数据名称 {data_name}')

    def eval_ood(self, fsood: bool = False, progress: bool = True):
        """评估 OOD 检测指标。

        Args:
            fsood (bool, optional):
                是否为 FSOOD 评估。默认为 False。
            progress (bool, optional):
                是否显示进度条。默认为 True。

        Returns:
            metrics_df (pd.DataFrame):
                OOD 评估指标的 DataFrame。
        """
        id_name = 'id' if not fsood else 'csid'
        task = 'ood' if not fsood else 'fsood'
        if self.metrics[task] is None:
            self.net.eval()

            # ID 分数
            if self.scores['id']['test'] is None:
                print(f'对 {self.id_name} 测试集进行推断...', flush=True)
                id_pred, id_conf, id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict['id']['test'], progress)
                self.scores['id']['test'] = [id_pred, id_conf, id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores['id']['test']

            if fsood:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores['csid'].keys()):
                    if self.scores['csid'][dataset_name] is None:
                        print(
                            f'对 {self.id_name} (cs) 测试集 [{i + 1}]: {dataset_name} 进行推断...',
                            flush=True)
                        temp_pred, temp_conf, temp_gt = \
                            self.postprocessor.inference(
                                self.net,
                                self.dataloader_dict['csid'][dataset_name],
                                progress)
                        self.scores['csid'][dataset_name] = [
                            temp_pred, temp_conf, temp_gt
                        ]

                    csid_pred.append(self.scores['csid'][dataset_name][0])
                    csid_conf.append(self.scores['csid'][dataset_name][1])
                    csid_gt.append(self.scores['csid'][dataset_name][2])

                csid_pred = np.concatenate(csid_pred)
                csid_conf = np.concatenate(csid_conf)
                csid_gt = np.concatenate(csid_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            # 加载 nearood 数据并计算 OOD 指标
            near_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                          ood_split='near',
                                          progress=progress)
            # 加载 farood 数据并计算 OOD 指标
            far_metrics = self._eval_ood([id_pred, id_conf, id_gt],
                                         ood_split='far',
                                         progress=progress)

            if self.metrics[f'{id_name}_acc'] is None:
                self.eval_acc(id_name)
            near_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
                                           len(near_metrics))
            far_metrics[:, -1] = np.array([self.metrics[f'{id_name}_acc']] *
                                          len(far_metrics))

            self.metrics[task] = pd.DataFrame(
                np.concatenate([near_metrics, far_metrics], axis=0),
                index=list(self.dataloader_dict['ood']['near'].keys()) +
                      ['nearood'] + list(self.dataloader_dict['ood']['far'].keys()) +
                      ['farood'],
                columns=['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC'],
            )
        else:
            print('评估已经完成！')

        with pd.option_context(
                'display.max_rows', None, 'display.max_columns', None,
                'display.float_format',
                '{:,.2f}'.format):  # 还可以指定更多选项
            print(self.metrics[task])

        return self.metrics[task]

    def _eval_ood(self,
                  id_list: List[np.ndarray],
                  ood_split: str = 'near',
                  progress: bool = True):
        """对 OOD 数据进行评估。

        Args:
            id_list (List[np.ndarray]):
                包含 ID 分数、置信度和标签的列表。
            ood_split (str, optional):
                OOD 分割方式（'near' 或 'far'）。默认为 'near'。
            progress (bool, optional):
                是否显示进度条。默认为 True。

        Returns:
            metrics_all (np.ndarray):
                包含所有评估指标的数组。
        """
        print(f'处理 {ood_split} ood 数据...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict['ood'][
            ood_split].items():
            if self.scores['ood'][ood_split][dataset_name] is None:
                print(f'对 {dataset_name} 数据集进行推断...', flush=True)
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                    self.net, ood_dl, progress)
                self.scores['ood'][ood_split][dataset_name] = [
                    ood_pred, ood_conf, ood_gt
                ]
            else:
                print(
                    '已对 '
                    f'{dataset_name} 数据集进行推断...',
                    flush=True)
                [ood_pred, ood_conf,
                 ood_gt] = self.scores['ood'][ood_split][dataset_name]

            ood_gt = -1 * np.ones_like(ood_gt)  # 硬性设为 -1 作为 ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'计算 {dataset_name} 数据集的指标...')
            ood_metrics = compute_all_metrics(conf, label, pred)
            metrics_list.append(ood_metrics)
            self._print_metrics(ood_metrics)

        print('计算平均指标...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        self._print_metrics(list(metrics_mean[0]))
        return np.concatenate([metrics_list, metrics_mean], axis=0) * 100

    def _print_metrics(self, metrics):
        """打印评估指标。

        Args:
            metrics (List[float]):
                包含 FPR@95、AUROC、AUPR_IN、AUPR_OUT 和 ACC 的指标列表。
        """
        [fpr, auroc, aupr_in, aupr_out, _] = metrics

        # 打印 ood 指标结果
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
            flush=True)
        print(u'\u2500' * 70, flush=True)
        print('', flush=True)

    def hyperparam_search(self):
        """进行超参数搜索。"""
        print('开始自动参数搜索...')
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)

        final_index = None
        for i, hyperparam in enumerate(hyperparam_combination):
            self.postprocessor.set_hyperparam(hyperparam)

            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['id']['val'])
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict['ood']['val'])

            ood_gt = -1 * np.ones_like(ood_gt)  # 硬性设为 -1 作为 ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]

            print('超参数: {}, auroc: {}'.format(hyperparam, auroc))
            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        print('最终超参数: {}'.format(
            self.postprocessor.get_hyperparam()))
        self.postprocessor.hyperparam_search_done = True

    def recursive_generator(self, list, n):
        """生成超参数组合的递归函数。

        Args:
            list (List[List[Any]]):
                包含所有超参数值的列表。
            n (int):
                超参数的数量。

        Returns:
            results (List[List[Any]]):
                所有超参数组合的列表。
        """
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
