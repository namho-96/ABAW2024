import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple

class Mlp(nn.Module):
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

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(
            self, dim, num_heads, use_cross, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        if use_cross:
            self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.use_cross = use_cross
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        output = self.attn(x)
        if self.use_cross:
            output = x[0] + self.drop_path1(self.norm1(output))
            output = x[0] + self.drop_path2(self.norm2(self.mlp(output)))
        else:
            output = output + self.drop_path1(self.norm1(output))
            output = output + self.drop_path2(self.norm2(self.mlp(output)))
        return output


class DeepMixAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.args = config
        hs1, hs2, hs3 = self.args.hidden_size
        self.model_arch = self.args.model_arch
        self.bs, self.sq, self.nf = self.args.batch_size, self.args.sq_len, self.args.num_features
        self.nh, self.dropout, self.droppath = self.args.num_head, self.args.dropout, self.args.droppath

        self.transformer = TransformerBlock(self.nf, self.nh, False, qkv_bias=True, drop=self.dropout, attn_drop=self.dropout, init_values=1e-5, drop_path=self.droppath)
        self.transformeria = TransformerBlock(self.nf, self.nh, True, qkv_bias=True, drop=self.dropout, attn_drop=self.dropout, init_values=1e-5, drop_path=self.droppath)
        self.transformerai = TransformerBlock(self.nf, self.nh, True, qkv_bias=True, drop=self.dropout, attn_drop=self.dropout, init_values=1e-5, drop_path=self.droppath)

        self.norm = nn.LayerNorm(self.nf * 2)
        self.norm2 = nn.LayerNorm(self.nf)
        self.mlp = nn.Linear(self.nf * 2, self.nf)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.conv1d = nn.Conv1d(self.nf, hs1, kernel_size=1, padding=0)  # 768 -> 256

        self.layers = nn.ModuleList()
        for layer_type in self.model_arch:
            if layer_type == "self":
                self.layers.append(self.transformer)
                self.layers.append(self.transformer)

            elif layer_type == "mix":
                self.layers.append(self.transformeria)
                self.layers.append(self.transformerai)

            elif layer_type == "forward":
                self.layers.append(self.transformeria)
                self.layers.append(self.transformerai)

            elif layer_type == 'expr':
                self.layers.append(nn.Linear(self.nf, self.args.num_classes))

            elif layer_type == 'au':
                self.layers.append(nn.Linear(self.nf, self.args.num_classes))

            elif layer_type == 'va':

                vhead = nn.Sequential(
                    nn.Linear(hs1, hs2),
                    nn.BatchNorm1d(hs2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hs2, hs3),
                    nn.BatchNorm1d(hs3),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hs3, 1),
                    nn.Tanh()
                )
                ahead = nn.Sequential(
                    nn.Linear(hs1, hs2),
                    nn.BatchNorm1d(hs2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hs2, hs3),
                    nn.BatchNorm1d(hs3),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hs3, 1),
                    nn.Tanh()
                )
                # Add VA head
                self.layers.append(vhead)
                self.layers.append(ahead)

        self.apply(self.init_weights)


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

    def forward(self, img, aud):
        img = F.interpolate(img, size=self.nf, mode='linear')  # 1024 -> 768
        aud = F.interpolate(aud, size=self.nf, mode='linear')  # 1024 -> 768
        layer_index = 0  # 실제로 사용된 레이어의 인덱스를 추적
        # Even - Video / Odd - Audio
        for layer_type in self.model_arch:
            if layer_index > len(self.layers):
                raise ValueError("Layer index out of range[index > model_arch]")

            if layer_type == "self":
                img_self = self.layers[layer_index](img)
                aud_self = self.layers[layer_index + 1](aud)
                layer_index += 2

            elif layer_type == "mix":
                temp_img, temp_aud = img.clone(), aud.clone()
                img = self.layers[layer_index]([temp_img, temp_aud])
                aud = self.layers[layer_index + 1]([temp_aud, temp_img])
                layer_index += 2  # type2의 경우 두 개의 레이어를 사용

            elif layer_type == "forward":
                # type3 처리: img와 aud의 조합
                img = self.layers[layer_index]([img, img_self])
                aud = self.layers[layer_index + 1]([aud, aud_self])
                layer_index += 2  # type3의 경우 한 개의 레이어를 사용

            elif layer_type == "neck":
                x = torch.cat((img, aud), 2)
                x = self.norm(x)  # B L (C*2)
                x = self.mlp(x)  # B L C
                x = self.transformer(x)
                x = self.norm2(x)

            elif layer_type == "expr":
                out = self.layers[layer_index](x)
                layer_index += 1
                return out

            elif layer_type == "au":
                out = self.layers[layer_index](x)
                layer_index += 1
                return out

            elif layer_type == "va":
                x = x.permute(0, 2, 1)  # [bs, nf, sq]
                x = self.conv1d(x)      # [bs, nf*, sq]
                x = x.permute(0, 2, 1)  # [bs, sq, nf*]
                x = torch.reshape(x, (self.bs * self.sq, -1))  # [bs * sq, nf*]

                vout = self.layers[layer_index](x)
                aout = self.layers[layer_index + 1](x)

                vout = vout.view(self.bs, self.sq, -1)
                aout = aout.view(self.bs, self.sq, -1)

                layer_index += 2
                return [vout, aout]





