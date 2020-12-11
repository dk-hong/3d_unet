import torch
import torch.nn as nn


def conv_block_3d(in_plane, middle_plane, out_plane, activation):
    return nn.Sequential(
        nn.Conv3d(in_plane, middle_plane, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(middle_plane),
        activation,
        nn.Conv3d(middle_plane, out_plane, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_plane),
        activation,
    )


def up_layer(in_plane, out_plane, activation=None, up_sample=False):
    if up_sample:
        return conv_trans_block_3d(in_plane, out_plane, activation)
    else:
        return nn.Upsample(scale_factor=2)


def conv_trans_block_3d(in_plane, out_plane, activation=None):
    return nn.Sequential(
        nn.ConvTranspose3d(in_plane, out_plane, kernel_size=3, stride=2, padding=1, output_padding=1),
        # nn.BatchNorm3d(out_plane),
        # activation,
    )


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters, up_sample=False):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.ReLU(inplace=True)

        # Down sampling
        self.conv_down_1 = conv_block_3d(self.in_dim, self.num_filters * 8, self.num_filters * 16, activation)
        self.conv_down_2 = conv_block_3d(self.num_filters * 16, self.num_filters * 16, self.num_filters * 32, activation)
        self.conv_down_3 = conv_block_3d(self.num_filters * 32, self.num_filters * 32, self.num_filters * 64, activation)

        # Pooling layer
        self.down = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # Bridge
        self.bridge = conv_block_3d(self.num_filters * 64, self.num_filters * 64, self.num_filters * 128, activation)

        # Up sampling
        self.up_1 = up_layer(self.num_filters * 128, self.num_filters * 128, activation, up_sample)
        self.up_2 = up_layer(self.num_filters * 64, self.num_filters * 64, activation, up_sample)
        self.up_3 = up_layer(self.num_filters * 32, self.num_filters * 32, activation, up_sample)

        self.conv_up_1 = conv_block_3d(self.num_filters * 192, self.num_filters * 64, self.num_filters * 64, activation)
        self.conv_up_2 = conv_block_3d(self.num_filters * 96, self.num_filters * 32, self.num_filters * 32, activation)
        self.conv_up_3 = conv_block_3d(self.num_filters * 48, self.num_filters * 16, self.num_filters * 16, activation)

        # Output
        self.out = nn.Conv3d(self.num_filters * 16, out_dim, kernel_size=3, stride=1, padding=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Down sampling
        conv_down_1 = self.conv_down_1(x)
        pool_1 = self.down(conv_down_1)

        conv_down_2 = self.conv_down_2(pool_1)
        pool_2 = self.down(conv_down_2)

        conv_down_3 = self.conv_down_3(pool_2)
        pool_3 = self.down(conv_down_3)

        # Bridge
        bridge = self.bridge(pool_3)

        # Up sampling
        up_1 = self.up_1(bridge)
        concat_1 = torch.cat([conv_down_3, up_1], dim=1)
        conv_up_1 = self.conv_up_1(concat_1)

        up_2 = self.up_2(conv_up_1)
        concat_2 = torch.cat([conv_down_2, up_2], dim=1)
        conv_up_2 = self.conv_up_2(concat_2)

        up_3 = self.up_3(conv_up_2)
        concat_3 = torch.cat([conv_down_1, up_3], dim=1)
        conv_up_3 = self.conv_up_3(concat_3)

        # Output
        out = self.out(conv_up_3)
        return self.sigmoid(out)

