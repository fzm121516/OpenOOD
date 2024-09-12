import numpy as np
from sklearn import metrics

# 计算所有的指标
def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)  # 设置打印精度为3位小数
    recall = 0.95  # 设置召回率阈值为0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)  # 计算AUROC, AUPR以及在给定召回率下的FPR

    accuracy = acc(pred, label)  # 计算准确率

    results = [fpr, auroc, aupr_in, aupr_out, accuracy]  # 将所有结果存储在列表中

    return results  # 返回结果列表

# 计算准确率
def acc(pred, label):
    ind_pred = pred[label != -1]  # 筛选出标签不为-1的预测值
    ind_label = label[label != -1]  # 筛选出标签不为-1的真实标签

    num_tp = np.sum(ind_pred == ind_label)  # 计算预测正确的数量
    acc = num_tp / len(ind_label)  # 计算准确率

    return acc  # 返回准确率

# 计算在给定召回率下的FPR和阈值
def fpr_recall(conf, label, tpr):
    gt = np.ones_like(label)  # 创建一个与标签相同形状的全1数组
    gt[label == -1] = 0  # 将标签为-1的位置设为0

    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)  # 计算ROC曲线
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]  # 找到满足召回率条件的最小FPR
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]  # 找到满足召回率条件的阈值
    return fpr, thresh  # 返回FPR和阈值

# 计算AUC和在给定召回率下的FPR
def auc_and_fpr_recall(conf, label, tpr_th):
    # 按照机器学习的惯例，我们将OOD样本视为正类
    ood_indicator = np.zeros_like(label)  # 创建一个与标签相同形状的全0数组
    ood_indicator[label == -1] = 1  # 将标签为-1的位置设为1

    # 在后处理器中，我们假设ID样本的"conf"值会大于OOD样本
    # 因此这里我们需要取反"conf"值
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)  # 计算取反后的ROC曲线
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]  # 找到满足召回率条件的最小FPR

    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(1 - ood_indicator, conf)  # 计算精确率-召回率曲线 (ID样本)
    precision_out, recall_out, thresholds_out = metrics.precision_recall_curve(ood_indicator, -conf)  # 计算精确率-召回率曲线 (OOD样本)

    auroc = metrics.auc(fpr_list, tpr_list)  # 计算AUROC
    aupr_in = metrics.auc(recall_in, precision_in)  # 计算ID样本的AUPR
    aupr_out = metrics.auc(recall_out, precision_out)  # 计算OOD样本的AUPR

    return auroc, aupr_in, aupr_out, fpr  # 返回AUROC、AUPR以及FPR

# 计算在给定FPR下的CCR
def ccr_fpr(conf, fpr, pred, label):
    ind_conf = conf[label != -1]  # 筛选出标签不为-1的置信度值
    ind_pred = pred[label != -1]  # 筛选出标签不为-1的预测值
    ind_label = label[label != -1]  # 筛选出标签不为-1的真实标签

    ood_conf = conf[label == -1]  # 筛选出标签为-1的置信度值

    num_ind = len(ind_conf)  # 计算ID样本的数量
    num_ood = len(ood_conf)  # 计算OOD样本的数量

    fp_num = int(np.ceil(fpr * num_ood))  # 计算给定FPR下的假阳性数量
    thresh = np.sort(ood_conf)[-fp_num]  # 找到FPR对应的阈值
    num_tp = np.sum((ind_conf > thresh) * (ind_pred == ind_label))  # 计算在阈值下预测正确的ID样本数量
    ccr = num_tp / num_ind  # 计算CCR

    return ccr  # 返回CCR

# 计算最小检测误差
def detection(ind_confidences,
              ood_confidences,
              n_iter=100000,
              return_data=False):
    # 计算最小检测误差
    Y1 = ood_confidences  # OOD样本的置信度
    X1 = ind_confidences  # ID样本的置信度

    start = np.min([np.min(X1), np.min(Y1)])  # 计算置信度的最小值
    end = np.max([np.max(X1), np.max(Y1)])  # 计算置信度的最大值
    gap = (end - start) / n_iter  # 计算迭代步长

    best_error = 1.0  # 初始化最小检测误差为1.0
    best_delta = None  # 初始化最优阈值为None
    all_thresholds = []  # 初始化所有阈值列表
    all_errors = []  # 初始化所有误差列表
    for delta in np.arange(start, end, gap):  # 在区间内遍历所有阈值
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))  # 计算在当前阈值下的TPR
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))  # 计算在当前阈值下的误差
        detection_error = (tpr + error2) / 2.0  # 计算检测误差

        if return_data:
            all_thresholds.append(delta)  # 记录当前阈值
            all_errors.append(detection_error)  # 记录当前误差

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)  # 更新最小检测误差
            best_delta = delta  # 更新最优阈值

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds  # 返回最小检测误差、最优阈值以及所有误差和阈值列表
    else:
        return best_error, best_delta  # 返回最小检测误差和最优阈值
