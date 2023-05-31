import torch
from torch import nn, einsum
import torch.nn.functional as F
#import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os, sys
import numpy as ny
from torch.nn import init
import torchvision.models as models
from collections import OrderedDict
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
from torch import nn

#*****************************************************************************************
#以下模型为Vision Transformer like CLIP ,vision model.
#*****************************************************************************************
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: list, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution[0]//patch_size)*(input_resolution[1]//patch_size) + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, :, :])
        x_local = x[:, 1:, :]
        if self.proj is not None:
            x_global = x[:, 0, :] @ self.proj

        return x_global,x_local

class Image_encoder_ViT(nn.Module):
    def __init__(self, backbone:str):
        super().__init__()
        if backbone == "ViT_B/32":
            self.backbone = VisionTransformer([256,128],32,768,12,12,768)
        elif backbone == "ViT_B/16":
            self.backbone = VisionTransformer([256, 128], 16, 768, 12, 12, 768)
        else:
            raise Exception("The model chosen is not existed")
    def forward(self, img):
        global_out, local_out = self.backbone(img)
        return global_out, local_out

class Image_decoder_ViT_projection(nn.Module):
    def __init__(self,img_dim:int,txt_dim:int,patch_dim:int):
        super().__init__()
        self.fc1 = nn.Linear(img_dim+txt_dim,patch_dim,bias=True)
        self.fc2 = nn.Linear(patch_dim, patch_dim, bias=True)
    def forward(self,img_embedding,txt_embedding):
        x = torch.cat((img_embedding, txt_embedding), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Image_decoder_ViT(nn.Module):
    def __init__(self,input_resolution: list, patch_size: int, width: int, layers: int, heads: int):
        super().__init__()
        output_dim = 3*(patch_size**2)
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution[0] // patch_size) * (input_resolution[1] // patch_size), width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self,img_embedding,txt_embedding):
        x = img_embedding + txt_embedding
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, :, :])
        x = x @ self.proj
        return x

#*****************************************************************************************
#以下模型为MResNet,vision model.
#*****************************************************************************************
class SE_block_mm(nn.Module):
    def __init__(self, image_dim, text_dim):
        super(SE_block_mm, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(image_dim+text_dim,image_dim)
        self.fc2 = nn.Linear(image_dim, image_dim)
        init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        init.kaiming_normal_(self.fc2.weight, mode='fan_out')

    def forward(self,image_feature,txt_embed):
        b,c,_,_ = image_feature.size()
        attention = self.avgpool(image_feature)
        attention = torch.flatten(attention, 1)
        attention = torch.cat((attention, txt_embed), dim=-1)
        attention = self.fc1(attention)
        attention = F.relu(attention)
        attention = self.fc2(attention)
        attention = F.sigmoid(attention).view(b,c,1,1)
        return image_feature*attention.expand_as(image_feature)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out+identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim_x: int,spacial_dim_y: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim_x * spacial_dim_y + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        return x.squeeze(0)

class Image_encoder_ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    def __init__(self, layers, output_dim, heads, input_resolution=[256,128], width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        #self.attnpool = AttentionPool2d(input_resolution[0] // 32,input_resolution[1] // 32, embed_dim, heads, output_dim)
        self.attnpool = None
        #self.probe_linearing = nn.Linear(768, 768)
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_proj = nn.Linear(embed_dim, output_dim)
        # self.maxpooling = nn.AdaptiveMaxPool2d((1, 1))
        # self.max_proj = nn.Linear(embed_dim, output_dim)
        self.initialize_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def initialize_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            #print(x.shape)
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        #feat = self.attnpool(x4)
        feat = self.avgpooling(x4)
        feat = torch.flatten(feat,1)
        feat = self.avg_proj(feat)
        #feat = self.probe_linearing(feat)
        #print(feat_set.shape)
        return feat,x1,x2,x3,x4

class Image_encoder_ModifiedResNetv2(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """
    def __init__(self, layers, output_dim, heads, input_resolution=[256,128], width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual layers
        self._inplanes = width*2  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution[0] // 32,input_resolution[1] // 32, embed_dim, heads, output_dim)
        self.initialize_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def initialize_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.maxpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        feat = self.attnpool(x4)
        return feat,x1,x2,x3,x4

class Image_decoder_ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=[256,128], width=64,text_embed_dim=768):
        super().__init__()
        base = Image_encoder_ModifiedResNet(layers, output_dim, heads, input_resolution, width)
        self.channelunity1 = nn.Conv2d(base.layer1[-1].conv3.out_channels, base.layer1[-1].conv3.out_channels,kernel_size=(1, 1))
        self.channelunity2 = nn.Conv2d(base.layer2[-1].conv3.out_channels, base.layer1[-1].conv3.out_channels,kernel_size=(1, 1))
        self.channelunity3 = nn.Conv2d(base.layer3[-1].conv3.out_channels, base.layer1[-1].conv3.out_channels,kernel_size=(1, 1))
        self.channelunity4 = nn.Conv2d(base.layer4[-1].conv3.out_channels, base.layer1[-1].conv3.out_channels,kernel_size=(1, 1))
        self.SE_block1 = SE_block_mm(base.layer1[-1].conv3.out_channels*4,text_embed_dim)
        self.SE_block2 = SE_block_mm(base.layer1[-1].conv3.out_channels, text_embed_dim)
        self.deconv1  = nn.ConvTranspose2d(base.layer1[-1].conv3.out_channels*4,base.layer1[-1].conv3.out_channels,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(base.layer1[-1].conv3.out_channels,3,kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self,x1,x2,x3,x4,txt_embed):
        c1 = self.channelunity1(x1)
        c2 = self.channelunity2(x2)
        c3 = self.channelunity3(x3)
        c4 = self.channelunity4(x4)

        y3 = F.interpolate(c4, scale_factor=2, mode='bilinear') + c3
        y2 = F.interpolate(y3, scale_factor=2, mode='bilinear') + c2
        y1 = F.interpolate(y2, scale_factor=2, mode='bilinear') + c1

        y2 = F.interpolate(y2, scale_factor=2, mode='bilinear')
        y3 = F.interpolate(y3, scale_factor=4, mode='bilinear')
        y4 = F.interpolate(c4, scale_factor=8, mode='bilinear')

        out = torch.cat([y1, y2, y3, y4], dim=1)
        out = self.SE_block1(out,txt_embed)
        out = self.deconv1(out)
        out = self.SE_block2(out,txt_embed)
        out = self.deconv2(out)

        return out

if __name__ == '__main__':
    model = AttentionPool2d(256 // 32,128 // 32, 1024, 8, 768)
    input = torch.randn([1,1024,8,4])
    embed,attention = model(input)
    print(embed.shape)
    #print(attention[:,:,1:].shape)
    attention = attention[:,:,1:]
    print(attention.shape)
    print(attention)
    attention = rearrange(attention,'b c (h w) -> b c h w',h=8)
    t = F.interpolate(attention, scale_factor=32, mode='bilinear')
    #print(attention[:,:,:])
    print(t)
    print(t.shape)

