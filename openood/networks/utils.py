import torch
import torch.backends.cudnn as cudnn

from .densenet import DenseNet3
from .lenet import LeNet
from .resnet18 import ResNet18
from .resnet18L import ResNet18L
from .wrn import WideResNet


def get_network(network_config):

    num_classes = network_config.num_classes

    if network_config.name == 'res18':
        net = ResNet18(num_classes=num_classes)

    elif network_config.name == 'res18L':
        net = ResNet18L(num_classes=num_classes)

    elif network_config.name == 'lenet':
        net = LeNet(num_classes=num_classes, num_channel=3)

    elif network_config.name == 'lenet_bw':
        net = LeNet(num_classes=num_classes, num_channel=1)

    elif network_config.name == 'wrn':
        net = WideResNet(depth=28,
                         widen_factor=10,
                         dropRate=0.0,
                         num_classes=num_classes)

    elif network_config.name == 'densenet':
        net = DenseNet3(depth=100,
                        growth_rate=12,
                        reduction=0.5,
                        bottleneck=True,
                        dropRate=0.0,
                        num_classes=num_classes)

    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.pretrained:
        net.load_state_dict(torch.load(network_config.checkpoint),
                            strict=False)
        print('Model Loading Completed!')

    if network_config.num_gpus > 1:
        net = torch.nn.DataParallel(net,
                                    device_ids=list(
                                        range(network_config.num_gpus)))

    if network_config.num_gpus > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True

    return net
