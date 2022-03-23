from torch import nn


def get_LeNet5():
    r"""
    参考《动手学深度学习》
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
    nn.Linear(84, 10))
    return net


class LeNet(nn.Module):
    r"""
    使用ReLU, BN等机制增强后的LeNet
    """
    def __init__(self, *args, **kwargs):
        super(LeNet, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        X = self.avgpool1(X)
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.relu2(X)
        X = self.avgpool2(X)
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.bn3(X)
        X = self.relu3(X)
        X = self.fc2(X)
        X = self.bn4(X)
        X = self.relu4(X)
        X = self.fc3(X)
        return X
