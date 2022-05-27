import torch
from torch import nn


def conv_block(in_channels, num_channels):
    r"""
    ResNet改良后的“批量规范化，激活，卷积”架构
    """
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))


r"""
稠密网络主要由2部分构成：稠密块（dense block）和过渡层（transition layer）。 前者定义如何连接输入和输出，而后者则控制通道数量，使其不会太复杂。
"""


class DenseBlock(nn.Module):
    r"""
    稠密层定义
    """
    def __init__(self, num_convs, in_channels, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        layers = []
        for i in range(num_convs):
            layers.append(conv_block(in_channels +
                          i*num_channels, num_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat([X, Y], dim=1)
        return X

def transition_block(in_channels, num_channels):
    r"""
    过渡层
    """
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

def get_DenseNet():
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    # 接下来，类似于ResNet使用的4个残差块，DenseNet使用的是4个稠密块。 与ResNet类似，我们可以设置每个稠密块使用多少个卷积层。 这里我们设成4
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密快之间添加一个转换层，使得输出通道减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2
    # 与ResNet类似，最后接上全局汇聚层和全连接层来输出结果。
    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10)
    )
    return net

if __name__ == '__main__':
    r"""
    在下面的例子中，我们定义一个有2个输出通道数为10的DenseBlock。 使用通道数为3的输入时，我们会得到通道数为的输出。 卷积块的通道数控制了输出通道数相对于输入通道数的增长，因此也被称为增长率（growth rate）。
    """
    blk = DenseBlock(2, 3, 10)
    X = torch.randn(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape)
    # 对上一个例子中稠密块的输出使用通道数为10的过渡层。 此时输出的通道数减为10，高和宽均减半
    blk = transition_block(23, 10)
    print(blk(Y).shape)
