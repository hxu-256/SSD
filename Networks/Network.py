import torch
import torch.nn as nn
import torch.nn.functional as F
from Networks.common import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_encoder(in_channels, out_channels, filter_size, downsample_mode, act_fun, pad, need_bias):
    return nn.Sequential(
        conv(in_channels, out_channels, filter_size, stride=2, bias=need_bias, pad=pad, downsample_mode=downsample_mode),
        act(act_fun),
        bn(out_channels),
        SDTrans(out_channels, out_channels),
        act(act_fun),
        bn(out_channels))


def build_skip(in_channels, out_channels, filter_skip_size, act_fun, pad, need_bias):
    return nn.Sequential(
        conv(in_channels, out_channels, filter_skip_size, bias=need_bias, pad=pad),
        act(act_fun),
        bn(out_channels),
        SDTrans(out_channels, out_channels),
        act(act_fun),
        bn(out_channels))


def build_decoder(in_channels, out_channels, filter_size, act_fun, pad, need_bias):
    return nn.Sequential(
        conv(in_channels, out_channels, filter_size, bias=need_bias, pad=pad),
        act(act_fun),
        bn(out_channels),
        SDTrans(out_channels, out_channels),
        act(act_fun),
        bn(out_channels))


class Network(nn.Module):
    def __init__(self, input_depth, skip_n33d, filter_size_down, downsample_mode, upsample_mode, skip_n11, skip_n33u, filter_size_up, num_scales, num_output_channels):
        super(Network, self).__init__()
        num_channels_down                         =     [skip_n33d] * num_scales
        num_channels_skip                         =     [skip_n11] * num_scales
        num_channels_up                           =     [skip_n33u] * num_scales
        filter_size_down                          =     [filter_size_down] * num_scales
        filter_size_up                            =     [filter_size_up] * num_scales
        downsample_mode                           =     [downsample_mode] * num_scales
        upsample_mode                             =     [upsample_mode] * num_scales
        filter_skip_size, pad, act_fun, need_bias =     3, 'reflection', 'LeakyReLU', True


        self.encoders, self.skips, self.decoders, self.ups = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(num_scales):
            in_channels, out_channels             =     input_depth if i == 0 else num_channels_down[i-1], num_channels_down[i]
            self.encoders.append(build_encoder(in_channels, out_channels, filter_size_down[i], downsample_mode[i], act_fun, pad, need_bias))
            skip_in_channels, skip_out_channels   = input_depth if i == 0 else num_channels_down[i-1], num_channels_skip[i]
            self.skips.append(build_skip(skip_in_channels, skip_out_channels, filter_skip_size, act_fun, pad, need_bias))
            decoder_in_channels, decoder_out_channels = num_channels_skip[i] + num_channels_up[min(i + 1, num_scales - 1)], num_channels_up[i]
            self.decoders.append(build_decoder(decoder_in_channels, decoder_out_channels, filter_size_up[i], act_fun, pad, need_bias))            
            self.ups.append(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        self.recon_head = conv(num_channels_up[-1], num_output_channels, 1, bias=need_bias, pad=pad)


    def forward(self, x, x_mean=None):
        nC = x.shape[1]
        if x_mean is None:
            x_mean = torch.mean(x, dim=1, keepdim=True).repeat(1, nC, 1, 1)
        else:
            x_mean = x_mean.repeat(1, nC, 1, 1)

        x = x - x_mean
        encodings, skips = [], []
        for i in range(len(self.encoders)):
            skips.append(self.skips[i](x))
            x = self.encoders[i](x)
            encodings.append(x)

        for i in reversed(range(len(self.decoders))):
            x = self.ups[i](x)
            x = F.interpolate(x, size=skips[i].shape[2:], mode='nearest')
            x = self.decoders[i](torch.cat([x, skips[i]], dim=1))
        recon = self.recon_head(x)
        return recon + x_mean



def Network_load(rank):
    im_net = Network(input_depth=rank, skip_n33d=48, filter_size_down=3, downsample_mode='stride',
                upsample_mode = 'nearest', skip_n11 = 48, skip_n33u = 48,
                filter_size_up = 3, num_scales = 3, num_output_channels = rank).to(device)
    return im_net