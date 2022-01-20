import torch, math
from torch import nn
from torch.nn.functional import max_pool1d


class TFILM(torch.nn.Module):

    def __init__(self, filters, n_block):
        super(TFILM, self).__init__()
        self.filters = filters
        self.n_block = n_block
        # self.lstm = nn.LSTM(input_size=filters, hidden_size=filters)
        self.lstm = nn.LSTM(input_size=filters, hidden_size=filters, bidirectional=True)

    def make_normalizer(self, x_in):
        """applies an lstm layer on top of x_in"""
        # input (-1, 4096, n_filters) to (-1, 32, n_filters)
        # output: (-1, 32, n_filters)
        x_in_down = max_pool1d(input=x_in, kernel_size=self.n_block)
        x_rnn = self.lstm(x_in_down.transpose(2, 1))
        #return x_rnn[0]
        return torch.divide(torch.add(x_rnn[0][:, :, :int(x_rnn[0].shape[-1]/2)], x_rnn[0][:, :, int(x_rnn[0].shape[-1]/2):]), 2)

    def apply_normalizer(self, x_in, x_norm):
        x_shape = x_in.shape
        n_steps = int(math.ceil(x_shape[2] / self.n_block))
        # reshape input into blocks
        x_in = torch.reshape(x_in, shape=(-1, n_steps, self.n_block, self.filters))
        x_norm = torch.reshape(x_norm, shape=(-1, n_steps, 1, self.filters))
        # multiply
        x_out = x_norm * x_in
        # return to original shape
        x_out = torch.reshape(x_out, shape=x_shape)
        return x_out

    def forward(self, x):
        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x


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


class TFILMUNet(nn.Module):


    def downsampl(self, in_ch: int, out_ch: int, kernel_sz: int, padding_mode: str, layer: int):

        if padding_mode == "same":
            pad_calc = int((-1 + kernel_sz + (kernel_sz - 1) * (self.drate - 1)) / 2)
        else: pad_calc = 0

        return nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_sz, padding=pad_calc,
                      dilation=self.drate),
            nn.MaxPool1d(kernel_size=self.pool_size, padding=0, stride=self.stride),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def bottleneck(self, in_ch: int, out_ch: int, kernel_sz: int, padding_mode: str):

        if padding_mode == "same":
            pad_calc = int((-1 + kernel_sz + (kernel_sz - 1) * (self.drate - 1)) / 2)
        else: pad_calc = 0

        return nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_sz, padding=pad_calc,
                      dilation=self.drate),
            nn.MaxPool1d(kernel_size=2, padding=0, stride=self.stride),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def upsampl(self, in_ch: int, out_ch: int, kernel_sz: int, padding_mode: str, layer: int):

        upscale = 2
        if padding_mode == "same":
            pad_calc = int((-1 + kernel_sz + (kernel_sz - 1) * (self.drate - 1)) / 2)
        else: pad_calc = 0

        return nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=upscale * out_ch, kernel_size=kernel_sz, padding=pad_calc,
                      dilation=self.drate),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            PixelShuffle1D(upscale_factor=upscale),
            #TFILM(filters=out_ch, n_block=32)#128 // (2 ** layer))
        )

    def lastconv(self, in_ch: int, out_ch: int, kernel_sz: int, padding_mode: str):

        upscale = 2
        if padding_mode == "same":
            pad_calc = int((-1 + kernel_sz + (kernel_sz - 1) * (self.drate - 1)) / 2)
        else: pad_calc = 0

        # convolution --> dropout --> relu --> Subpixel shuffle --> stacking
        return nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=2 * out_ch, kernel_size=kernel_sz, padding=pad_calc,
                      dilation=self.drate),
            PixelShuffle1D(upscale_factor=upscale),
        )

    def __init__(self):
        super(TFILMUNet, self).__init__()
        self.flatten = nn.Flatten()
        self.stride = 2
        self.pool_size = 2
        self.drate = 2
        '''
        FROM PAPER:
        n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
        n_blocks = [ 128, 64, 32, 16, 8]
        n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
        downsampling_l = []
        '''

        # Network Layers
        # downsampling
        self.down1 = self.downsampl(1, 128, 65, "same", 0)
        self.tfilm_d1 = TFILM(filters=128, n_block=128 // (2 ** 0))
        self.down2 = self.downsampl(128, 256, 33, "same", 1)
        self.tfilm_d2 = TFILM(filters=256, n_block=128 // (2 ** 1))
        self.down3 = self.downsampl(256, 512, 17, "same", 2)
        self.tfilm_d3 = TFILM(filters=512, n_block=128 // (2 ** 2))
        self.down4 = self.downsampl(512, 512, 9,  "same", 3)
        self.tfilm_d4 = TFILM(filters=512, n_block=128 // (2 ** 3))
        # bottleneck
        self.bottle = self.bottleneck(512, 512, 9, "same")
        self.tfilm_b = TFILM(filters=512, n_block=128 // (2 ** 3))
        # upsampling
        self.up4 = self.upsampl(512, 512, 9, "same", 3)
        self.tfilm_u4 = TFILM(filters=512, n_block=128 // (2 ** 3))
        self.up3 = self.upsampl(1024, 256, 17, "same", 2)
        self.tfilm_u3 = TFILM(filters=256, n_block=128 // (2 ** 2))
        self.up2 = self.upsampl(768, 128, 33, "same", 1)
        self.tfilm_u2 = TFILM(filters=128, n_block=128 // (2 ** 1))
        self.up1 = self.upsampl(384, 1, 65, "same", 0)
        self.tfilm_u1 = TFILM(filters=1, n_block=128 // (2 ** 0))
        # last conv
        self.last = self.lastconv(129, 1, 9, "same")


    def forward(self, X):

        x = self.down1(X)
        conv1 = self.tfilm_d1(x)
        x = self.down2(x)
        conv2 = self.tfilm_d2(x)
        x = self.down3(conv2)
        conv3 = self.tfilm_d3(x)
        x = self.down4(conv3)
        conv4 = self.tfilm_d4(x)

        x = self.bottle(conv4)
        x = self.tfilm_b(x)

        x = self.up4(x)
        x = self.tfilm_u4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.up3(x)
        x = self.tfilm_u3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up2(x)
        x = self.tfilm_u2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up1(x)
        x = self.tfilm_u1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.last(x)
        x = torch.add(x, X)

        return x