import torch
import torch.nn as nn


class normalize(torch.nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        self.a = None
        self.b = None

    def forward(self, x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        self.a = 2.0 / (x_max + x_min)
        self.b = -(x_max - x_min) / (x_max + x_min)
        return self.a * x + self.b


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )

    def forward(self, x):
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        channel_att = self.fc(avg_pool).view(x.size(0), x.size(1), 1, 1, 1)
        return x * channel_att.sigmoid()


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        combined = torch.cat((max_pool, avg_pool), dim=1)
        spatial_att = self.conv(combined)
        return x * spatial_att.sigmoid()


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class UNet(nn.Module):
    def __init__(self, channel_list):
        super().__init__()

        N = len(channel_list)

        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv3d(
                    channel_list[i], channel_list[i + 1], kernel_size=3, padding=1
                )
                for i in range(0, N - 1)
            ]
        )

        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv3d(
                    channel_list[N - 1 - i] + channel_list[N - 1 - i],
                    channel_list[N - 2 - i],
                    kernel_size=3,
                    padding=1,
                )
                for i in range(0, N - 1)
            ]
        )

        self.filter_layers = torch.nn.ModuleList(
            [nn.Conv3d(2, 1, kernel_size=3, padding=1) for i in range(3)]
        )

        self.act = nn.SiLU()

    def forward(self, x):
        x0 = x.detach().clone()
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            h.append(x)

        for i, l in enumerate(self.up_layers):
            x = torch.cat((x, h.pop()), dim=1)
            x = self.act(l(x))

        for i, l in enumerate(self.filter_layers):
            x = torch.cat((x, x0), dim=1)
            x = self.act(l(x))

        return x


class UNet_attention(nn.Module):
    def __init__(self, channel_list):
        super().__init__()

        N = len(channel_list)

        self.down_layers = torch.nn.ModuleList(
            [
                nn.Conv3d(
                    channel_list[i], channel_list[i + 1], kernel_size=3, padding=1
                )
                for i in range(0, N - 1)
            ]
        )

        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv3d(
                    channel_list[N - 1 - i] + channel_list[N - 1 - i],
                    channel_list[N - 2 - i],
                    kernel_size=3,
                    padding=1,
                )
                for i in range(0, N - 1)
            ]
        )

        self.filter_layers = torch.nn.ModuleList(
            [nn.Conv3d(2, 1, kernel_size=3, padding=1) for i in range(3)]
        )

        self.attention_layers = torch.nn.ModuleList(
            [
                CBAM(in_channels=channel_list[i + 1], reduction_ratio=1)
                for i in range(0, N - 1)
            ]
        )

        self.act = nn.SiLU()

    def forward(self, x):
        x0 = x.detach().clone()
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))
            h.append(self.attention_layers[i](x))

        for i, l in enumerate(self.up_layers):
            x = torch.cat((x, h.pop()), dim=1)
            x = self.act(l(x))

        for i, l in enumerate(self.filter_layers):
            x = torch.cat((x, x0), dim=1)
            x = self.act(l(x))

        return x
