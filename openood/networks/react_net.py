import torch.nn as nn

class ReactNet(nn.Module):
    def __init__(self, backbone):
        # 初始化 ReactNet 类，接受一个 backbone 网络作为参数
        super(ReactNet, self).__init__()
        self.backbone = backbone  # 将 backbone 存储为网络的一个成员变量

    def forward(self, x, return_feature=False, return_feature_list=False):
        # 前向传播方法
        try:
            # 尝试调用 backbone 的 forward 方法
            return self.backbone(x, return_feature, return_feature_list)
        except TypeError:
            # 如果 backbone 的 forward 方法不支持 return_feature_list 参数，则只使用 return_feature 参数
            return self.backbone(x, return_feature)

    def forward_threshold(self, x, threshold):
        # 使用阈值进行前向传播，返回经过修正的分类结果
        _, feature = self.backbone(x, return_feature=True)  # 获取 backbone 的特征
        feature = feature.clip(max=threshold)  # 对特征进行裁剪，最大值不超过阈值 threshold
        feature = feature.view(feature.size(0), -1)  # 将特征展平，以便送入全连接层
        logits_cls = self.backbone.get_fc_layer()(feature)  # 通过全连接层计算分类 logits
        return logits_cls  # 返回分类 logits

    def get_fc(self):
        # 获取 backbone 的全连接层权重和偏置
        fc = self.backbone.fc  # 访问 backbone 的全连接层
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()  # 返回全连接层的权重和偏置，并转为 numpy 数组
