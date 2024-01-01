import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder


class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim, dropout_p=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

        # 1x1 conv to fuse multi-scale output from encoder
        self.layers = nn.ModuleList([nn.Conv2d(chans, embed_dim, (1, 1))
                                     for chans in reversed(in_channels)])
        self.linear_fuse = nn.Conv2d(embed_dim * len(self.layers), embed_dim, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(embed_dim, eps=1e-5)

        # 1x1 conv to get num_class channel predictions
        self.linear_pred = nn.Conv2d(self.embed_dim, num_classes, kernel_size=(1, 1))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        feature_size = x[0].shape[2:]

        # project each encoder stage output to H/4, W/4
        x = [layer(xi) for layer, xi in zip(self.layers, reversed(x))]
        x = [F.interpolate(xi, size=feature_size, mode='bilinear', align_corners=False)
             for xi in x[:-1]] + [x[-1]]

        # concatenate project output and use 1x1
        # convs to get num_class channel output
        x = self.linear_fuse(torch.cat(x, dim=1))
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.linear_pred(x)
        return x


if __name__ == "__main__":
    # test here
    x = torch.randn(size=(8, 3, 512, 1024))
    in_channels = 3
    encoder = Encoder(in_chans=in_channels, embed_dims=(64, 128, 320, 512),
                                num_heads=(1, 2, 5, 8), depths=(3, 4, 18, 3),
                                sr_ratios=(8, 4, 2, 1), dropout_p=0.0, drop_path_p=0.1)
    enc_output = encoder(x)

    # decoder block
    # Since the shape of enc_output is (64, 128, 320, 512)
    # These will be the in_channels
    # In cityscapes => 19 classes were there
    decoder_output = Decoder(in_channels=(64, 128, 320, 512),
                                    num_classes=19, embed_dim=256)
    print( decoder_output.shape )