import torch
import torch.nn as nn
from functools import partial

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FixEmbed(nn.Module):
    def __init__(self, in_c=1056, embed_dim=64, norm_layer=None):
        super().__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).transpose(1, 2).flatten(2) 
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,    
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        q=self.q(x.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k=self.k(x.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v=self.v(x.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class CCS_block(nn.Module):
    def __init__(self,dim,reduction,norm_layer,channel_reduction=False):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(dim, int(dim/reduction)),
                                nn.ReLU())
        self.channel_reduction=channel_reduction
        self.th_embedding=nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, 1),
            nn.Sigmoid())
        self.norm1 = norm_layer(dim)
        self.th_act_layer=nn.ReLU()
    def forward(self, x,cluster_center,alpha):
        x=self.norm1(x)
        cos_sim_numerator=x @ x.transpose(-2, -1)  
        x_norm=torch.norm(x,p=2,keepdim=True,dim=-1)
        cos_sim_denominator=x_norm@x_norm.transpose(-2, -1)
        cos_sim=cos_sim_numerator/(cos_sim_denominator+1e-8)

        density=torch.sum(cos_sim,dim=-1)#[8, 100]
        density_max=torch.max(density,dim=-1,keepdim=True)[0]
        density_min=torch.min(density,dim=-1,keepdim=True)[0]
        density=(density-density_min)/(density_max-density_min+1e-8)

        density_th=self.th_embedding(density)*alpha
        density=self.th_act_layer(density-density_th)
        density_thresholded=density/(torch.sum(density,dim=1,keepdim=True)+1e-8)

        distance_map=x-cluster_center.repeat(1,x.shape[1],1)
        shift_vector=distance_map*density_thresholded.unsqueeze(-1)
        shift_vector_avg=torch.mean(shift_vector,dim=1,keepdim=True)

        if self.channel_reduction==True:
            cluster_center=self.proj(cluster_center+shift_vector_avg)
            return cluster_center
        else:
            return cluster_center+shift_vector_avg

class CCG(nn.Module):
    def __init__(self,dim,reduction,norm_layer):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(dim, int(dim/reduction)),
                                nn.ReLU())
        self.norm1 = norm_layer(dim)
    
    def forward(self, x):
        x=self.norm1(x)
        B,N,C=x.shape
        cos_sim_numerator=x @ x.transpose(-2, -1)  
        x_norm=torch.norm(x,p=2,keepdim=True,dim=-1)
        cos_sim_denominator=x_norm@x_norm.transpose(-2, -1)
        cos_sim=cos_sim_numerator/(cos_sim_denominator+1e-8)

        density=torch.sum(cos_sim,dim=-1)#[8, 100]
        MAX_mask=torch.max(density,dim=1,keepdim=True)[0].repeat(1,N)
        MAX_mask=(density==MAX_mask).int().unsqueeze(-1).repeat(1,1,C)  
        cluster_center=torch.sum(x*MAX_mask,dim=1,keepdim=True)  
        cluster_center=self.proj(cluster_center)
        return cluster_center
        

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 channel_reduction=2,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm0 = norm_layer(dim)
        self.proj_reduction=nn.Conv2d(dim, int(dim/channel_reduction), kernel_size=1, stride=1)
        self.norm1 = norm_layer(int(dim/channel_reduction))        
        self.attn = Attention(int(dim/channel_reduction),num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(int(dim/channel_reduction))
        
    def forward(self, x):
        
        x=self.proj_reduction(self.norm0(x).transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(-1,-2) 
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Model(nn.Module):
    def __init__(self, in_c=96, num_classes=1, depth=4, num_heads=1, qkv_bias=True,
                 qk_scale=None, fix_embed_dim=64,fix_num=14, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=FixEmbed):
        super(Model, self).__init__()

        self.num_classes = num_classes
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        channel_reduction=2
        self.depth=depth
        self.fix_embed = embed_layer(in_c=in_c, embed_dim=fix_embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  
        embed_dim=[int(fix_embed_dim*fix_num/(channel_reduction**(x))) for x in range(depth)] 
        self.CCG =CCG(embed_dim[0],channel_reduction,norm_layer)

        for i in range(depth):
            block= Block(dim=embed_dim[i], num_heads=num_heads,channel_reduction=channel_reduction, qkv_bias=qkv_bias,
                          qk_scale=qk_scale,drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, 
                          drop_path_ratio=dpr[i],norm_layer=norm_layer)
            CCS=CCS_block(dim=int(embed_dim[i]/channel_reduction),norm_layer=norm_layer,reduction=channel_reduction,channel_reduction=(i!=(depth-1)))
            setattr(self, f"stage_{i + 1}", block)
            setattr(self, f"CCS_{i + 1}", CCS)
        self.norm = norm_layer(int(embed_dim[-1]/channel_reduction))
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(embed_dim[-1]/channel_reduction), num_classes) if num_classes > 0 else nn.Identity(),
            nn.Sigmoid())
        
        self.apply(_init_vit_weights)

    def forward(self, x,alpha=1):
        fix_embed = self.fix_embed(x)  
        cluster_center0=self.CCG(fix_embed) 
        
        stage1 = getattr(self, f"stage_{1}")
        CCS1 = getattr(self, f"CCS_{1}")
        x1 = stage1(fix_embed)
        cluster_center1=CCS1(x1,cluster_center0,alpha) 

        stage2 = getattr(self, f"stage_{2}")
        CCS2 = getattr(self, f"CCS_{2}")
        x2 = stage2(x1)
        cluster_center2=CCS2(x2,cluster_center1,alpha)

        stage3 = getattr(self, f"stage_{3}")
        CCS3 = getattr(self, f"CCS_{3}")
        x3 = stage3(x2)
        cluster_center3=CCS3(x3,cluster_center2,alpha)

        stage4 = getattr(self, f"stage_{4}")
        CCS4 = getattr(self, f"CCS_{4}")
        x4 = stage4(x3)
        cluster_center4=CCS4(x4,cluster_center3,alpha)

        x = self.norm(cluster_center4) 
        x = self.head(x) 

        return x





