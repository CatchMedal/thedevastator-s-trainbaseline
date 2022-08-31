"""
UNext Backbone  - EfficientNet
"""
import torch
from torch import nn
from torchvision.models import efficientnet_v2_l
import torch.nn.functional as F
from fastai.vision.all import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class UneXt50(nn.Module):
    def __init__(self, stride=1, **kwargs):
        super().__init__()
        # encoder
        m = efficientnet_v2_l()
        
        self.enc0, self.enc1, self.enc2, self.enc3, self.enc4, self.enc5, self.enc6, self.enc7\
            = m.features[0], m.features[1], m.features[2], m.features[3], m.features[4], m.features[5], m.features[6], m.features[7]
        # aspp with customized dilatations
        self.aspp = ASPP(640, 256, out_c=512, dilations=[
                         stride*1, stride*2, stride*3, stride*4])
        self.drop_aspp = nn.Dropout2d(0.5)
        # decoder
        # UnetBlock ( 이전 블록의 채널, skip-connection 채널, output 채널)
        self.dec4, self.dec3, self.dec2, self.dec1 = \
            UnetBlock(512, 192, 256),  UnetBlock(256, 96, 128), UnetBlock(128, 64, 64), UnetBlock(64, 32, 32)

        self.fpn = FPN([512,256,128,64], [16]*4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer( 32+16*4, 1, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.aspp(enc7)
        dec4 = self.dec4(self.drop_aspp(enc8), enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc0)
        x = self.fpn([enc8, dec4, dec3, dec2], dec1)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x


# split the model to encoder and decoder for fast.ai
def split_layers(m): return [list(m.enc0.parameters())+list(m.enc1.parameters()) +
                             list(m.enc2.parameters())+list(m.enc3.parameters()) +
                             list(m.enc4.parameters())+list(m.enc5.parameters()) +
                             list(m.enc6.parameters())+list(m.enc7.parameters()),
                             list(m.aspp.parameters())+list(m.dec4.parameters()) +
                             list(m.dec3.parameters())+list(m.dec2.parameters()) +
                             list(m.dec1.parameters())+list(m.fpn.parameters()) +
                             list(m.final_conv.parameters())]


class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
             nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch*2),
             nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
             for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs: list, last_layer):
        hcs = [F.interpolate(c(x), scale_factor=2**(len(self.convs)-i), mode='bilinear')
               for i, (c, x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class UnetBlock(Module):
    def __init__(self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False,
                 self_attention: bool = False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = nf if nf is not None else max(up_in_c//2, 32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
            [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4)
             for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                         nn.Conv2d(inplanes, mid_c, 1,
                                                   stride=1, bias=False),
                                         nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False),
                                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[
                           2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
