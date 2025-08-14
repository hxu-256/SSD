from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)
    


class Transformer_layer(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        self.dim = dim

    def forward(self, x_in, mask=None):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h*w, c)                              # Size: [b, h*w, c]
        q_inp = self.to_q(x)                                     # Size: [b, h*w, dim_head * heads]
        k_inp = self.to_k(x)                                     # Size: [b, h*w, dim_head * heads]
        v_inp = self.to_v(x)                                     # Size: [b, h*w, dim_head * heads]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp, k_inp, v_inp))
        q = q.transpose(-2, -1)                                  # Size: [b, dim_head, heads, h*w]
        k = k.transpose(-2, -1)                                  # Size: [b, dim_head, heads, h*w]
        v = v.transpose(-2, -1)                                  # Size: [b, dim_head, heads, h*w]
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))                         # A = K^T*Q
        attn = attn.softmax(dim=-1)                              # Size: [b, dim_head, heads, heads]
        x = attn @ v                                             # Size: [b, dim_head, heads, h*w]
        x = x.permute(0, 3, 1, 2)                                # Size: [b, h*w, dim_head, heads]
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)  # Size: [b, h*w, dim_head*heads]
        out_c = self.proj(x).view(b, h, w, c)                    # Size: [b, h*w, c]
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c * out_p
        return out



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False))

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)



class Transformer_block(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                Transformer_layer(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out



def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            #L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            L.append(Transformer_block(dim=in_channels, dim_head=int(in_channels/8), heads=8))
        elif t == 'D':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)



class SAA_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SAA_Block, self).__init__()
        self.spa_avg_pool = nn.AdaptiveAvgPool2d(8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()

        y_mean = self.spa_avg_pool(x)
        y_mean = y_mean.view(b, c, -1)
        y_mean = (y_mean - torch.min(y_mean)) / (torch.max(y_mean) - torch.min(y_mean))
        y_mean = torch.bmm(y_mean, y_mean.permute(0, 2, 1))
        y_mean = torch.mean(y_mean, 2).view(b, c, 1, 1)

        y = self.avg_pool(x * y_mean).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class TransBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, mode='CLBDLB', reduction=16, negative_slope=0.2):
        super(TransBlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.ca = SAA_Block(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res+x)
        return res
