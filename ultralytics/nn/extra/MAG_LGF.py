import torch
import torch.nn as nn
import torch.nn.functional as F # <--- 确保导入 F
import math

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_pool = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_pool)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class StructureAwareMultiScaleContextBranch(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = max(c1 // 2, 32)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.dw5 = nn.Conv2d(c_, c_, kernel_size=5, padding=2, groups=c_, bias=False)
        self.dw9 = nn.Conv2d(c_, c_, kernel_size=9, padding=4, groups=c_, bias=False)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(c_ * 6, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y_dw5 = self.dw5(x)
        y_dw9 = self.dw9(x)
        out = torch.cat((x, y1, y2, y3, y_dw5, y_dw9), dim=1)
        return self.cv2(out)


class LocalGlobalFusionBranch(nn.Module):
    def __init__(self, dim, mlp_ratio=4, lk_size=9, pool_size=2, num_heads=8):
        super().__init__()
        self.dim = dim
        self.pool_size = pool_size
        hidden_features = int(dim * mlp_ratio)

        self.norm1 = nn.BatchNorm2d(dim)
        self.local_agg = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=lk_size, padding=lk_size // 2, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, kernel_size=1),
        )
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size)
        self.norm3 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.upsample = nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=False)
        self.norm4 = nn.BatchNorm2d(dim)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, kernel_size=1),
        )

    def forward(self, x_in):
        B, C, H, W = x_in.shape
        shortcut_x = x_in
        x = self.local_agg(self.norm1(x_in))
        x = x + shortcut_x

        shortcut_y = x
        y = self.ffn1(self.norm2(x))
        y = y + shortcut_y

        shortcut_z = y
        y_pooled = self.pool(y)
        B_p, C_p, H_p, W_p = y_pooled.shape
        y_tokens = y_pooled.flatten(2).permute(0, 2, 1)
        y_tokens_norm = self.norm3(y_tokens)
        attn_output, _ = self.mha(y_tokens_norm, y_tokens_norm, y_tokens_norm)
        attn_img = attn_output.permute(0, 2, 1).reshape(B, C, H_p, W_p)

        z = self.upsample(attn_img)
        if z.shape[-2:] != shortcut_z.shape[-2:]:
            z = F.interpolate(z, size=shortcut_z.shape[-2:], mode='bilinear', align_corners=False)

        z = z + shortcut_z
        shortcut_out = z
        x_out = self.ffn2(self.norm4(z))
        x_out = x_out + shortcut_out
        return x_out


class MAG_LGF(nn.Module):

    def __init__(self, c1, c2, reduce_factor=2, debug=False):
        super().__init__()
        self.debug = debug
        assert c1 > 0 and c2 > 0
        c_mid = max(c1 // reduce_factor, 64)

        self.adapt_in = nn.Identity()
        self.reduce_conv = Conv(c1, c_mid, 1, 1)
        self.cbam = CBAM(c_mid)
        self.samscb = StructureAwareMultiScaleContextBranch(c_mid, c_mid)
        self.lgfb = LocalGlobalFusionBranch(c_mid) # 现在使用已更正的模块


        self.target_in_channels = c1
        self.target_out_channels = c2

        self.fuse_conv = nn.Conv2d(c_mid, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        if self.debug:
            print(f"[Debug] Input shape to OptimizedFusion: {x.shape}")
        x = self.adapt_in(x)
        x_reduced = self.reduce_conv(x)

        x_cbam = self.cbam(x_reduced)
        x_samscb = self.samscb(x_cbam)
        x_lgfb = self.lgfb(x_cbam)

        x_fused = torch.cat((x_samscb, x_lgfb), dim=1)
        # 若未定义融合卷积层，则自动创建 (从2*c2 → c2)
        if not hasattr(self, "fuse_conv") or self.fuse_conv.in_channels != x_fused.shape[1]:
            self.fuse_conv = nn.Conv2d(x_fused.shape[1], self.target_out_channels, 1, 1).to(x.device)
            self.bn = nn.BatchNorm2d(self.target_out_channels)
            self.act = nn.SiLU()


        out = self.act(self.bn(self.fuse_conv(x_fused)))

        if self.debug:
            print(f"[Debug] Output shape from OptimizedFusion: {out.shape}")
        return out