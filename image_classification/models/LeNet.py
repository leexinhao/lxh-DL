from torch import nn


def get_LeNet(input_shape=(28, 28), output_dim=10):
    r"""
    输入为28x28图像
    输出为10维向量（10分类）
    输入输出写死是因为这是古董模型了，也就在Mnist或Fashion Mnist上玩玩
    我们对原始模型做了一点小改动，去掉了最后一层的高斯激活。除此之外，这个网络与最初的LeNet-5一致。
    """
    net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, output_dim))
    return net
