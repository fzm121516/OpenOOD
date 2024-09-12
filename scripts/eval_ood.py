import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle
import collections
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.evaluation_api import Evaluator

from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.csi_net import CSINet
from openood.networks.udg_net import UDGNet
from openood.networks.cider_net import CIDERNet
from openood.networks.npos_net import NPOSNet
from openood.networks.palm_net import PALMNet
from openood.networks.t2fnorm_net import T2FNormNet

# 更新字典d中的值，如果u中的键在d中存在，则更新其值；如果u中的键在d中不存在，则添加该键值对。
def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# 创建参数解析器对象
parser = argparse.ArgumentParser()
# 添加必须的参数'--root'
parser.add_argument('--root', required=True)
# 添加可选参数'--postprocessor'，默认值为'msp'
parser.add_argument('--postprocessor', default='msp')
# 添加可选参数'--id-data'，类型为字符串，默认值为'cifar10'
parser.add_argument(
    '--id-data',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100', 'aircraft', 'cub', 'imagenet200'])
# 添加可选参数'--batch-size'，类型为整数，默认值为200
parser.add_argument('--batch-size', type=int, default=200)
# 添加布尔类型的可选参数'--save-csv'，用于控制是否保存结果为CSV文件
parser.add_argument('--save-csv', action='store_true')
# 添加布尔类型的可选参数'--save-score'，用于控制是否保存评分
parser.add_argument('--save-score', action='store_true')
# 添加布尔类型的可选参数'--fsood'，用于控制是否进行fsood评估
parser.add_argument('--fsood', action='store_true')
# 解析命令行参数
args = parser.parse_args()

# 获取参数'root'的值
root = args.root

# 指定一个实现的后处理器名称，例如'openmax', 'msp', 'temp_scaling', 'odin'等
postprocessor_name = args.postprocessor

# 定义一个字典，用于存储不同数据集的类别数量
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}
# 定义一个字典，用于存储不同数据集对应的模型架构
MODEL = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224,
}

# 尝试从字典中获取参数'id-data'对应的数据集类别数量和模型架构
try:
    num_classes = NUM_CLASSES[args.id_data]
    model_arch = MODEL[args.id_data]
except KeyError:
    raise NotImplementedError(f'ID dataset {args.id_data} is not supported.')

# 假设根文件夹包含每次训练运行对应的子文件夹，例如s0, s1, s2
# 如果使用OpenOOD进行训练，会自动创建这种结构
if len(glob(os.path.join(root, 's*'))) == 0:
    raise ValueError(f'No subfolders found in {root}')

# 迭代每次训练运行的子文件夹
all_metrics = []
for subfolder in sorted(glob(os.path.join(root, 's*'))):
    # 如果存在预设的后处理器，加载它
    if os.path.isfile(
            os.path.join(subfolder, 'postprocessors',
                         f'{postprocessor_name}.pkl')):  # 后处理器文件路径
        with open(
                os.path.join(subfolder, 'postprocessors',
                             f'{postprocessor_name}.pkl'), 'rb') as f:
            postprocessor = pickle.load(f)  # 加载后处理器
    else:
        postprocessor = None

    # 根据用户提供的后处理器名称加载预训练模型
    if postprocessor_name == 'conf_branch':
        net = ConfBranchNet(backbone=model_arch(num_classes=num_classes),
                            num_classes=num_classes)
    elif postprocessor_name == 'godin':
        backbone = model_arch(num_classes=num_classes)
        net = GodinNet(backbone=backbone,
                       feature_size=backbone.feature_size,
                       num_classes=num_classes)
    elif postprocessor_name == 'rotpred':
        net = RotNet(backbone=model_arch(num_classes=num_classes),
                     num_classes=num_classes)
    elif 'csi' in root:
        backbone = model_arch(num_classes=num_classes)
        net = CSINet(backbone=backbone,
                     feature_size=backbone.feature_size,
                     num_classes=num_classes)
    elif 'udg' in root:
        backbone = model_arch(num_classes=num_classes)
        net = UDGNet(backbone=backbone,
                     num_classes=num_classes,
                     num_clusters=1000)
    elif postprocessor_name in ['cider', 'reweightood']:
        backbone = model_arch(num_classes=num_classes)
        net = CIDERNet(backbone,
                       head='mlp',
                       feat_dim=128,
                       num_classes=num_classes)
    elif postprocessor_name == 'npos':
        backbone = model_arch(num_classes=num_classes)
        net = NPOSNet(backbone,
                      head='mlp',
                      feat_dim=128,
                      num_classes=num_classes)
    elif postprocessor_name == 'palm':
        backbone = model_arch(num_classes=num_classes)
        net = PALMNet(backbone,
                      head='mlp',
                      feat_dim=128,
                      num_classes=num_classes)
        postprocessor_name = 'mds'
    elif postprocessor_name == 't2fnorm':
        backbone = model_arch(num_classes=num_classes)
        net = T2FNormNet(backbone=backbone, num_classes=num_classes)
    else:
        net = model_arch(num_classes=num_classes)

    # 加载预训练模型权重
    net.load_state_dict(
        torch.load(os.path.join(subfolder, 'best.ckpt'), map_location='cpu'))
    net.cuda()
    net.eval()

    # 创建评估器对象
    evaluator = Evaluator(
        net,
        id_name=args.id_data,  # 目标ID数据集名称
        data_root=os.path.join(ROOT_DIR, 'data'),
        config_root=os.path.join(ROOT_DIR, 'configs'),
        preprocessor=None,  # 使用默认预处理
        postprocessor_name=postprocessor_name,
        postprocessor=postprocessor,  # 用户可以传递自己的后处理器
        batch_size=args.batch_size,  # 对于某些方法，结果可能会受到批量大小的轻微影响
        shuffle=False,
        num_workers=8)

    # 如果存在预计算的评分文件，加载它们
    if os.path.isfile(
            os.path.join(subfolder, 'scores', f'{postprocessor_name}.pkl')):
        with open(
                os.path.join(subfolder, 'scores', f'{postprocessor_name}.pkl'),
                'rb') as f:
            scores = pickle.load(f)
        update(evaluator.scores, scores)  # 更新评估器的评分
        print('Loaded pre-computed scores from file.')

    # 保存后处理器以便将来重复使用
    if hasattr(evaluator.postprocessor, 'setup_flag'
               ) or evaluator.postprocessor.hyperparam_search_done is True:
        pp_save_root = os.path.join(subfolder, 'postprocessors')
        if not os.path.exists(pp_save_root):
            os.makedirs(pp_save_root)

        if not os.path.isfile(
                os.path.join(pp_save_root, f'{postprocessor_name}.pkl')):
            with open(os.path.join(pp_save_root, f'{postprocessor_name}.pkl'),
                      'wb') as f:
                pickle.dump(evaluator.postprocessor, f,
                            pickle.HIGHEST_PROTOCOL)

    # 进行OOD评估，并将结果存储在all_metrics列表中
    metrics = evaluator.eval_ood(fsood=args.fsood)
    all_metrics.append(metrics.to_numpy())

    # 如果指定了保存评分，保存评分文件
    if args.save_score:
        score_save_root = os.path.join(subfolder, 'scores')
        if not os.path.exists(score_save_root):
            os.makedirs(score_save_root)
        with open(os.path.join(score_save_root, f'{postprocessor_name}.pkl'),
                  'wb') as f:
            pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)

# 计算多次训练运行的平均指标
all_metrics = np.stack(all_metrics, axis=0)
metrics_mean = np.mean(all_metrics, axis=0)
metrics_std = np.std(all_metrics, axis=0)

# 创建最终的指标列表
final_metrics = []
for i in range(len(metrics_mean)):
    temp = []
    for j in range(metrics_mean.shape[1]):
        temp.append(u'{:.2f} \u00B1 {:.2f}'.format(metrics_mean[i, j],
                                                   metrics_std[i, j]))
    final_metrics.append(temp)
df = pd.DataFrame(final_metrics, index=metrics.index, columns=metrics.columns)

# 如果指定了保存CSV文件，保存结果为CSV文件
if args.save_csv:
    saving_root = os.path.join(root, 'ood' if not args.fsood else 'fsood')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)
    df.to_csv(os.path.join(saving_root, f'{postprocessor_name}.csv'))
else:
    print(df)
