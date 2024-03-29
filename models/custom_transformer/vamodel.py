import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class CustomMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        
        B, N, C = x1.shape
        q = x1.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ResPostBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = CustomMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x = self.attn(x)
        x = x + self.drop_path1(self.norm1(x))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class VAmodel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.arch = config.model_arch
        self.bs, self.sq, self.nf, self.nh = config.batch_size, config.sq_len, config.num_features, config.num_head
        self.do = config.dropout

        self.layers = nn.ModuleList()

        self.transformeria = ResPostBlock(self.nf, self.nh, qkv_bias=True, init_values=1e-5)
        self.transformerai = ResPostBlock(self.nf, self.nh, qkv_bias=True, init_values=1e-5)

        self.norm = nn.LayerNorm(self.nf * 2)
        self.mlp = nn.Linear(self.nf * 2, self.nf)

        self.transformer = ResPostBlock(self.nf, self.nh, qkv_bias=True, init_values=1e-5)

        self.norm2 = nn.LayerNorm(self.nf)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(config.num_features, config.num_classes) if config.num_classes > 0 else nn.Identity()

        hs1, hs2, hs3 = config.hidden_size
        self.feat_fc = nn.Conv1d(self.nf, hs1, kernel_size=1, padding=0)     # 768 -> 256
        self.tanh = nn.Tanh()

        self.vhead = nn.Sequential(
            nn.Linear(hs1, hs2),
            nn.BatchNorm1d(hs2),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(hs2, hs3),
            nn.BatchNorm1d(hs3),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(hs3, 1),
            nn.Tanh()
        )
        self.ahead = nn.Sequential(
            nn.Linear(hs1, hs2),
            nn.BatchNorm1d(hs2),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(hs2, hs3),
            nn.BatchNorm1d(hs3),
            nn.GELU(),
            nn.Dropout(self.do),
            nn.Linear(hs3, 1),
            nn.Tanh()
        )

        self.apply(self.init_weights)

    def av_va_fusion(self, img, aud):
        aud = F.interpolate(aud, size=768, mode='linear')   # 1024 -> 768
        trans_va = self.transformeria([img, aud])
        trans_av = self.transformerai([aud, img])

        x = torch.cat((trans_va, trans_av), 2)
        x = self.norm(x)  # B L (C*2)
        x = self.mlp(x)  # B L C

        x = self.transformer([x, x])
        x = self.norm2(x)

        return x


    def deep_mix(self, img, aud):
        aud = F.interpolate(aud, size=768, mode='linear')  # 1024 -> 768

        for trans_type in self.config.layers:
            if trans_type == 'mix':
                pass
            elif trans_type == 'self':
                pass

        img = self.transformer([img, img])
        img = self.transformer([img, img])

        aud = self.transformer([aud, aud])
        aud = self.transformer([aud, aud])

        trans_va = self.transformeria([img, aud])
        trans_av = self.transformerai([aud, img])

        mix_va = self.transformeria([trans_va, trans_av])
        mix_av = self.transformerai([trans_av, trans_va])

        mix_va = self.transformeria([mix_va, mix_av])
        mix_av = self.transformerai([mix_av, mix_va])

        x = torch.cat((mix_va, mix_av), 2)
        x = self.norm(x)  # B L (C*2)
        x = self.mlp(x)  # B L C

        x = self.transformer([x, x])
        x = self.norm2(x)

        return x


    def forward(self, img, aud):
        x = self.deep_mix(img, aud)     # [bs, sq, nf]
        # x = self.av_va_fusion(img, aud)  # [bs, sq, nf]

        x = x.permute(0, 2, 1)              # [bs, nf, sq]
        x = self.feat_fc(x)                 # [bs, nf*, sq]
        # x = self.LeakyReLU(x)
        # x = self.tanh(x)
        x = x.permute(0, 2, 1)              # [bs, sq, nf*]
        x = torch.reshape(x, (self.bs * self.sq, -1))    # [bs * sq, nf*]

        vid = self.vhead(x)
        aud = self.ahead(x)
        vid = vid.view(self.bs, self.sq, -1)
        aud = aud.view(self.bs, self.sq, -1)

        return [vid, aud]

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)