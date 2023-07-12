import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def _make_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        out = torch.cat(features, dim=1)
        return out

'''
# 创建DenseBlock模块
in_channels = 1
growth_rate = 8
num_layers = 4
dense_block = DenseBlock(in_channels, growth_rate, num_layers)

# 构造输入张量
input_tensor = torch.randn(3, 1, 128, 128)

# 前向传播
output_tensor = dense_block(input_tensor)

print(output_tensor.shape)
[3, 33, 128, 128]

dense_block = DenseBlock(1, 8, 5)
[3, 41, 128, 128]
dense_block = DenseBlock(1, 16, 4)
[3, 65, 128, 128]
dense_block = DenseBlock(1, 8, 6)
[3, 49, 128, 128]
dense_block = DenseBlock(1, 16, 5)
[3, 81, 128, 128]


'''