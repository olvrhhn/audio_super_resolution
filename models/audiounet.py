import torch
from torch import nn


class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class AudioUNet(nn.Module):
    def __init__(self):
        super(AudioUNet, self).__init__()
        self.flatten = nn.Flatten()
        self.stride = 4
        self.pool_size = 4
        self.drate = 2
        '''
        FROM PAPER:
        n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
        n_blocks = [ 128, 64, 32, 16, 8]
        n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
        downsampling_l = []
        '''

        def downsampl(opts, in_ch: int, out_ch: int, kernel_sz: int, padding_mode: str):

            if padding_mode == "same":
                pad_calc = int((-1 + kernel_sz + (kernel_sz - 1) * (self.drate - 1)) / 2)
            else:
                pad_calc = 0
            # convolution --> batch norm --> relu
            return nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_sz, padding=pad_calc,
                          dilation=opts.drate),
                nn.MaxPool1d(kernel_size=self.pool_size, padding=0, stride=opts.stride),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(num_features=out_ch),
            )

        def bottleneck(opts, in_ch: int, out_ch: int, kernel_sz: int, padding_mode: str):

            if padding_mode == "same":
                pad_calc = int((-1 + kernel_sz + (kernel_sz - 1) * (opts.drate - 1)) / 2)
            else:
                pad_calc = 0
            # convolution --> dropout --> relu --> Subpixel shuffle --> stacking
            return nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_sz, padding=pad_calc,
                          dilation=opts.drate),
                nn.MaxPool1d(kernel_size=4, padding=0, stride=opts.stride),
                nn.Dropout(p=0.5),
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=out_ch),
            )

        def upsampl(opts, in_ch: int, out_ch: int, kernel_sz: int, padding_mode: str):

            upscale = 4
            if padding_mode == "same":
                pad_calc = int((-1 + kernel_sz + (kernel_sz - 1) * (opts.drate - 1)) / 2)
            else:
                pad_calc = 0
            # convolution --> dropout --> relu --> Subpixel shuffle --> stacking
            return nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=upscale * out_ch, kernel_size=kernel_sz, padding=pad_calc,
                          dilation=opts.drate),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                PixelShuffle1D(upscale_factor=upscale),
                nn.BatchNorm1d(num_features=out_ch),
            )

        def lastconv(opts, in_ch: int, out_ch: int, kernel_sz: int, padding_mode: str):

            upscale = 4
            if padding_mode == "same":
                pad_calc = int((-1 + kernel_sz + (kernel_sz - 1) * (opts.drate - 1)) / 2)
            else:
                pad_calc = 0
            # convolution --> dropout --> relu --> Subpixel shuffle --> stacking
            return nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=2 * out_ch, kernel_size=kernel_sz, padding=pad_calc,
                          dilation=opts.drate),
                PixelShuffle1D(upscale_factor=upscale),
            )

        # Network Layers
        # downsampling
        self.down1 = downsampl(self, 1,   128, 65, "same")
        self.down2 = downsampl(self, 128, 256, 33, "same")
        self.down3 = downsampl(self, 256, 512, 17, "same")
        self.down4 = downsampl(self, 512, 512, 9,  "same")
        # bottleneck
        self.bottle = bottleneck(self, 512, 512, 9, "same")
        # upsampling
        self.up4 = upsampl(self, 512, 512, 9, "same")
        self.up3 = upsampl(self, 1024, 256, 17, "same")
        self.up2 = upsampl(self, 768, 128, 33, "same")
        self.up1 = upsampl(self, 384, 1, 65, "same")
        # last conv
        self.lastconv = lastconv(self, 129, 2, 9, "same")


    def forward(self, X):

        conv1 = self.down1(X)
        conv2 = self.down2(conv1)
        conv3 = self.down3(conv2)
        conv4 = self.down4(conv3)

        x = self.bottle(conv4)

        x = self.up4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.up3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.lastconv(x)
        x = torch.add(x, X)

        return x