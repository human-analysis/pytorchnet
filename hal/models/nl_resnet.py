import torch
import torch.nn as nn
from .non_learnable_modules import NonLearnableLayer

_TRACK_RUNNING_STATS = False

"""ResNet with non-learnable layers"""

__all__ = ['nlresnet18']

class ConvLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, norm_layer=None,
                 non_learnable=False, binary=False, sparsity=0.99, depthwise=False):
        super(ConvLayer, self).__init__()
        if non_learnable:
            groups = in_planes if depthwise else 1
            conv = NonLearnableLayer(in_planes, out_planes, kernel_size, stride, padding, groups,
                                     sparsity=sparsity, binary=binary)
        else:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False)

        modules = [conv]

        if norm_layer is not None:
            modules.append(norm_layer(num_features=out_planes, track_running_stats=_TRACK_RUNNING_STATS))

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv_block(x)

    def get_weight(self):
        return self.conv_block[0].weight


def get_blocks(model):
    if isinstance(model, ConvLayer):
        blocks = [model]
    else:
        blocks = []
        for child in model.children():
            blocks += get_blocks(child)
    return blocks


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64, norm_layer=None, non_learnable=False,
                 binary=False, sparsity=0.99, depthwise=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvLayer(
            inplanes, planes, kernel_size=3, padding=1, stride=stride, norm_layer=norm_layer,
            non_learnable=non_learnable, binary=binary,
            sparsity=sparsity, depthwise=depthwise)
        self.conv2 = ConvLayer(
            planes, planes, kernel_size=3, padding=1, norm_layer=norm_layer,
            non_learnable=non_learnable, binary=binary, sparsity=sparsity, depthwise=depthwise)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.len = 3 if downsample is not None else 2
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, in_chs=3,
                 width_per_group=64, norm_layer=None, trainable_weights=False,
                 non_learnable=False, binary=False, sparsity=0.99, depthwise=False):
        super(ResNet, self).__init__()
        self.non_learnable = non_learnable
        self.binary = binary
        self.sparsity = sparsity
        self.depthwise = depthwise
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_chs, self.out_chs, self.wid_mult = in_chs, 512, 2
        self.inplanes = 64
        self.base_width = width_per_group
        self.conv1 = ConvLayer(self.in_chs, self.inplanes, kernel_size=7, stride=2, padding=3, norm_layer=norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * self.wid_mult, layers[0])
        self.layer2 = self._make_layer(block, 128 * self.wid_mult, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * self.wid_mult, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.out_chs, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(self.out_chs, num_classes)
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = ConvLayer(self.inplanes, planes, kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.base_width, norm_layer,
                            non_learnable=self.non_learnable, binary=self.binary,
                            sparsity=self.sparsity, depthwise=self.depthwise))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                base_width=self.base_width, norm_layer=norm_layer,
                                non_learnable=self.non_learnable,
                                binary=self.binary, sparsity=self.sparsity, depthwise=self.depthwise))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        return out

    def get_blocks(self):
        return get_blocks(self)

    def get_weights(self):
        return [block.get_weight() for block in self.get_blocks()]

    def get_block(self, depth):
        return self.get_blocks()[depth]

    def get_weight(self, depth):
        return self.get_weights()[depth]


def nlresnet18(**kwargs):
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


if __name__ == '__main__':
    from torchprofile import profile_macs

    model = resnet18(num_classes=10, non_learnable=True, binary=False, sparsity=0.99, depthwise=True)
    print(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_count_full = sum(p.numel() for p in model.parameters())

    data = torch.rand(1, 3, 64, 64)
    out = model(data)
    flops = profile_macs(model, data) / 1e6
    print(param_count)
    print(param_count_full)
    print(flops)
