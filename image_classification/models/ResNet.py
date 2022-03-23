from torch import nn


class Residual(nn.Module):
    def __init__(self, in_channels, num_channels,
                 use_1x1conv=False, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels,
                      kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels)
        )
        if use_1x1conv:
            self.p2 = nn.Conv2d(in_channels, num_channels,
                                kernel_size=1, stride=stride)
        else:
            self.p2 = None
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = self.p1(X)
        if self.p2 is not None:
            X = self.p2(X)
        Y += X
        return self.relu(Y)


def resnet_block(in_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:  # 第一个残差块不减小特征图大小
            blk.append(Residual(in_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(
            *resnet_block(256, 512, 2), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        return self.classifier(x)


if __name__ == "__main__":
    import torch
    blk = Residual(3, 3)
    print(blk)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)
    blk = Residual(3, 6, use_1x1conv=True, stride=2)
    print(blk)
    print(blk(X).shape)
