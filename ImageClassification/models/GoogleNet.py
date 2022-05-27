import torch
from torch import nn

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU()
        )
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.p1(x)
        x2 = self.p2(x)
        x3 = self.p3(x)
        x4 = self.p4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)

class GoogleNet(nn.Module):
    r"""
    简化10分类版本
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                Inception(256, 128, (128, 192), (32, 96), 64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                Inception(512, 160, (112, 224), (24, 64), 64),
                                Inception(512, 128, (128, 256), (24, 64), 64),
                                Inception(512, 112, (144, 288), (32, 64), 64),
                                Inception(528, 256, (160, 320), (32, 128), 128),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                Inception(832, 384, (192, 384), (48, 128), 128),
                                nn.AdaptiveAvgPool2d((1, 1)))
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 10)
        )
        
    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        return self.classifer(X)

if __name__ == "__main__":
    net = GoogleNet()
    X = torch.rand(size=(1, 1, 96, 96))
    for layer in net.children():
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)   