import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out += identity
        return out


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=32, dropout=False, pooling_size=2):
        super(ResUNet, self).__init__()

        if dropout:
            dropout_layer = nn.Dropout(0.1)
        else:
            dropout_layer = nn.Identity()

        self.init_path = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            ResidualBlock(features, features, kernel_size=3, padding=1),
            ResidualBlock(features, features, kernel_size=3, padding=1),
            ResidualBlock(features, features, kernel_size=3, padding=1)
        )
        self.shortcut0 = nn.Conv2d(features, features, kernel_size=1)

        self.down1 = nn.Sequential(
            nn.BatchNorm2d(features),
            nn.Conv2d(features, features * 2, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResidualBlock(features * 2, features * 2, kernel_size=3, padding=1),
            ResidualBlock(features * 2, features * 2, kernel_size=3, padding=1),
            ResidualBlock(features * 2, features * 2, kernel_size=3, padding=1)
        )
        self.shortcut1 = nn.Conv2d(features * 2, features * 2, 1)

        self.down2 = nn.Sequential(
            nn.BatchNorm2d(features * 2),
            nn.Conv2d(features * 2, features * 4, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResidualBlock(features * 4, features * 4, kernel_size=3, padding=1),
            ResidualBlock(features * 4, features * 4, kernel_size=3, padding=1),
            ResidualBlock(features * 4, features * 4, kernel_size=3, padding=1)
        )
        self.shortcut2 = nn.Conv2d(features * 4, features * 4, 1)

        self.down3 = nn.Sequential(
            nn.BatchNorm2d(features * 4),
            nn.Conv2d(features * 4, features * 8, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer,
            ResidualBlock(features * 8, features * 8, kernel_size=3, padding=1),
            ResidualBlock(features * 8, features * 8, kernel_size=3, padding=1),
            ResidualBlock(features * 8, features * 8, kernel_size=3, padding=1),
            dropout_layer
        )

        self.up3 = nn.Sequential(
            ResidualBlock(features * 8, features * 8, kernel_size=3, padding=1),
            ResidualBlock(features * 8, features * 8, kernel_size=3, padding=1),
            ResidualBlock(features * 8, features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        )

        self.up2 = nn.Sequential(
            ResidualBlock(features * 4, features * 4, kernel_size=3, padding=1),
            ResidualBlock(features * 4, features * 4, kernel_size=3, padding=1),
            ResidualBlock(features * 4, features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        )

        self.up1 = nn.Sequential(
            ResidualBlock(features * 2, features * 2, kernel_size=3, padding=1),
            ResidualBlock(features * 2, features * 2, kernel_size=3, padding=1),
            ResidualBlock(features * 2, features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ConvTranspose2d(features * 2, features, kernel_size=pooling_size, stride=pooling_size, bias=False),
            nn.ReLU(),
            dropout_layer
        )

        self.out_path = nn.Sequential(
            ResidualBlock(features, features, kernel_size=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x0 = self.init_path(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x2_up = self.up3(x3)
        x1_up = self.up2(x2_up + self.shortcut2(x2))
        x0_up = self.up1(x1_up + self.shortcut1(x1))
        x_out = self.out_path(x0_up + self.shortcut0(x0))

        return torch.sigmoid(x_out)
