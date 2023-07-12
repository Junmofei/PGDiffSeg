import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import self_attention1


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.shortcut = nn.Sequential()
        # # 经过处理后的x要与x的维度相同(尺寸和深度)
        # # 如果不相同，需要添加卷积+BN来变换为同一维度
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion * planes,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion * planes)
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = F.relu(out)  # 4.22/21：47
        return out

class ResNet2(nn.Module):
    def __init__(self, block, base_channels, rate, num_blocks, num_classes=2, in_ch=1, size=128):
        super(ResNet2, self).__init__()
        self.in_planes = base_channels
        channels = [base_channels * i for i in rate]
        in_planes = [base_channels] + channels[:-1]
        out_planes = channels
        self.avg_pool_out = size//(2**len(rate))

        self.conv1 = nn.Conv2d(in_ch, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.layers = nn.ModuleList(self._make_layer(block, base_channels, base_channels, num_blocks[0], stride=1))
        for  i in range(len(channels)): # 一层一个下采样
            self.layers.append(self._make_layer(block, in_planes[i], out_planes[i], num_blocks[i], stride=2))

        self.linear = nn.Linear(channels[-1], num_classes)
        self.out_conv = nn.Sequential(
                                     nn.Conv2d(channels[-2], 1, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Sigmoid()
                                     )
        self.gradients=None

    def _make_layer(self, block, in_plane, out_plane, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_plane, out_plane, stride))
            in_plane = out_plane
        return nn.Sequential(*layers)

    @torch.no_grad()
    def get_map(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        return out
        
    @torch.no_grad()
    def get_map2(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers[:-1]:
            out = layer(out)
        return out
    
    def save_gradient(self, grad):
        # print('save grad')
        self.gradients = grad
    
    def cam(self, x):
        a = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            a = layer(a)
        # print(out.shape, self.avg_pool_out)
        a.register_hook(self.save_gradient)  # 最后一个特征图
        out = F.avg_pool2d(a, self.avg_pool_out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, a
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers[:-1]:
            out = layer(out)
        # print(out.shape, self.avg_pool_out)
        dice_map = self.out_conv(out)
        out = self.layers[-1](out)
        
        out = F.avg_pool2d(out, self.avg_pool_out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, dice_map

class ResNet3(nn.Module):
    def __init__(self, block, base_channels, rate, num_blocks, num_classes=2, in_ch=1, size=128):
        super(ResNet3, self).__init__()
        self.in_planes = base_channels
        channels = [base_channels * i for i in rate]
        in_planes = [base_channels] + channels[:-1]
        out_planes = channels
        self.avg_pool_out = size//(2**len(rate))

        self.conv1 = nn.Conv2d(in_ch, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.layers = nn.ModuleList(self._make_layer(block, base_channels, base_channels, num_blocks[0], stride=1))
        for  i in range(len(channels)): # 一层一个下采样
            self.layers.append(self._make_layer(block, in_planes[i], out_planes[i], num_blocks[i], stride=2))

        self.linear = nn.Linear(channels[-1], num_classes)
        self.out_conv = nn.Sequential(
                                     nn.Conv2d(channels[-1], 1, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Sigmoid()
                                     )
    
    def _make_layer(self, block, in_plane, out_plane, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_plane, out_plane, stride))
            in_plane = out_plane
        return nn.Sequential(*layers)

    @torch.no_grad()
    def get_map(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        # print(out.shape, self.avg_pool_out)
        dice_map = self.out_conv(out)
        
        out = F.avg_pool2d(out, self.avg_pool_out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, dice_map
        
def get_resnet2(args):
    base_channels = args.res_base_channels
    rate = args.resnet_rate
    size = args.img_size
    block = BasicBlock if args.res_block is None else eval(args.res_block)
    num_blocks = [2]*len(rate) if args.num_blocks is None else eval(args.num_blocks)
    print('image size:', size)
    print('base_channels:', base_channels)
    print('rate:', rate)
    print('block:', block)
    print('num_blocks:', num_blocks)
    return ResNet2(block, base_channels, rate, num_blocks, size=size, num_classes=args.classes)

def get_resnet3(args):
    base_channels = args.res_base_channels
    rate = args.resnet_rate
    size = args.img_size
    block = BasicBlock if args.res_block is None else eval(args.res_block)
    num_blocks = [2]*len(rate) if args.num_blocks is None else eval(args.num_blocks)
    print('image size:', size)
    print('base_channels:', base_channels)
    print('rate:', rate)
    print('block:', block)
    print('num_blocks:', num_blocks)
    return ResNet3(block, base_channels, rate, num_blocks, size=size, num_classes=args.classes)

def classify_model(args):
    if args.classify_model == 'resnet18':
        exit()
        # print('model: resnet18')
        # model = ResNet(BasicBlock, [2, 2, 2, 2], args.use_sigmoid, num_classes=args.classes, in_ch=1, size=args.img_size)  # ressnet18
    elif args.classify_model == 'adapt_resnet2':
        model = get_resnet2(args)
    elif args.classify_model == 'adapt_resnet3':
        model = get_resnet3(args)
    else:
        raise RuntimeError('error model name', args.model)
    return model

