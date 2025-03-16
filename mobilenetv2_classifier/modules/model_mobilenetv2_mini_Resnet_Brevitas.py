import torch
import torch.nn as nn 
import math

from brevitas.core.restrict_val import RestrictValueType

from brevitas.nn import QuantConv2d
from brevitas.nn import QuantLinear
from brevitas.nn import QuantReLU
from brevitas.nn import QuantIdentity
from brevitas.nn import TruncAvgPool2d
from brevitas.quant import Int32Bias

from brevitas.quant import TruncTo8bit

from .common import CommonIntWeightPerChannelQuant
from .common import CommonIntWeightPerTensorQuant
from .common import CommonUintActQuant
from .common import CommonIntActQuant

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride, weight_bit_width, act_bit_width):

    return nn.Sequential(
        QuantConv2d(
            in_channels=inp,
            out_channels=oup,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width),
        nn.BatchNorm2d(oup),
        QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width),
    )
            
        


def conv_1x1_bn(inp, oup, weight_bit_width, act_bit_width):

    return nn.Sequential(
        QuantConv2d(
            in_channels=inp,
            out_channels=oup,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width),
        nn.BatchNorm2d(oup),
        QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            return_quant_tensor=True),
    )


class InvertedBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, weight_bit_width, act_bit_width, shared_quant):
        super(InvertedBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if self.identity:
            self.pw_linear_quant = shared_quant.act_quant
            # Add quantizer only needed if block is a Resnet one
            self.quant_identity_out = QuantIdentity(
                act_quant=CommonIntActQuant,
                bit_width=act_bit_width)
        else:
            self.pw_linear_quant = CommonIntActQuant    

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                QuantConv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(hidden_dim),
                QuantReLU(
                    act_quant=CommonUintActQuant,
                    bit_width=act_bit_width),
                # pw-linear
                QuantConv2d(
                    in_channels=hidden_dim,
                    out_channels=oup,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(oup),
                QuantIdentity(
                    act_quant=self.pw_linear_quant,
                    bit_width=act_bit_width),
                )
        else:
            self.conv = nn.Sequential(
                # pw
                QuantConv2d(
                    in_channels=inp,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(hidden_dim),
                QuantReLU(
                    act_quant=CommonUintActQuant,
                    bit_width=act_bit_width),
                # dw
                QuantConv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(hidden_dim),
                QuantReLU(
                    act_quant=CommonUintActQuant,
                    bit_width=act_bit_width),
                # pw-linear
                QuantConv2d(
                    in_channels=hidden_dim,
                    out_channels=oup,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(oup),
                QuantIdentity(
                    act_quant=self.pw_linear_quant,
                    bit_width=act_bit_width),
                )

        

    def forward(self, x):
        if self.identity:
            x = x + self.conv(x)
            x = self.quant_identity_out(x)
            return x 
        else:
            return self.conv(x)

    # Return the Quantizer of the Output, to use it in next block if Resnet Connection is applied
        # If current block is a Resnet one, return Add Quantizer
        # If current block is not Resnet, return pw-linear quantizer
    def get_shared_quant(self):
        if self.identity:
            return self.quant_identity_out
        else:
            return self.conv[-1] 



class MobileNetV2_MINI_RESNET(nn.Module):
    def __init__(self, num_classes=2, 
                 width_mult=1., 
                 in_bit_width=8,
                 weight_bit_width=4,
                 act_bit_width=4):
        super(MobileNetV2_MINI_RESNET, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  8, 1, 1],
            [2,  16, 2, 2],
            [2,  24, 2, 2],
            [4,  32, 3, 2],
            [2,  64, 2, 1],
        ]
        
        # Input 224x224x3
        layers = [
                QuantIdentity( # for Q1.7 input format -> sign.7bits
                act_quant = CommonIntActQuant,
                bit_width = in_bit_width,
                min_val = -1.0,
                max_val = 1.0 - 2.0 ** (-7),
                narrow_range = False, 
                restrict_scaling_type = RestrictValueType.POWER_OF_TWO)
        ]
        
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers.append(conv_3x3_bn(3, input_channel, 2, weight_bit_width, act_bit_width))
        
        # building inverted residual blocks
        block = InvertedBlock
        shared_quant = None
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                new_block = block(input_channel, output_channel, s if i == 0 else 1, t, weight_bit_width, act_bit_width, shared_quant)
                layers.append(new_block)
                shared_quant = new_block.get_shared_quant()
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        
        # building last several layers
        output_channel = _make_divisible(128 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 128
        self.conv = conv_1x1_bn(input_channel, output_channel, weight_bit_width, act_bit_width)
        
        self.avgpool = TruncAvgPool2d(
                kernel_size=14,  
                trunc_quant=TruncTo8bit,
                float_to_int_impl_type='FLOOR')
        self.classifier = QuantLinear(
            output_channel,
            num_classes,
            bias=True,
            bias_quant=Int32Bias,
            weight_quant=CommonIntWeightPerTensorQuant,
            weight_bit_width=8)

        
    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



                



                
