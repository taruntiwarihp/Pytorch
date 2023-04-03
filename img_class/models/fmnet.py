import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)     
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0., use_postln=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )              
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        # context aggreation
        ctx_all = 0 
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*self.gates[:,self.focal_level:]

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q*self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)
        
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

class FocalNetBlock(nn.Module):

    def __init__(self, dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0., 
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                    focal_level=1, focal_window=3,
                    use_layerscale=False, layerscale_value=1e-4, 
                    use_postln=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level

        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(dim, proj_drop=drop, focal_window=focal_window, focal_level=self.focal_level, use_postln=use_postln)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.H = None
        self.W = None

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x

        # Focal Modulation
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.modulation(x).view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample=None, focal_level=1, focal_window=1, 
                 use_conv_embed=False, 
                 use_layerscale=False, layerscale_value=1e-4, use_postln=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # build blocks
        self.blocks = nn.ModuleList([
            FocalNetBlock(
                dim=dim, 
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio, 
                drop=drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window, 
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value,
                use_postln=use_postln, 
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, 
                patch_size=2, 
                in_chans=dim, 
                embed_dim=out_dim, 
                use_conv_embed=use_conv_embed, 
                norm_layer=norm_layer, 
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x)

        if self.downsample is not None:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W        
        return x, Ho, Wo

class PatchEmbed(nn.Module):

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=None, is_stem=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)        
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W

class FMNet(nn.Module):

    def __init__(self, 
                img_size=224, 
                patch_size=4, 
                in_chans=3, 
                num_classes=1000,
                embed_dim=96, 
                depths=[2, 2, 6, 2], 
                mlp_ratio=4., 
                drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, 
                patch_norm=True,
                focal_levels=[2, 2, 2, 2], 
                focal_windows=[3, 3, 3, 3], 
                use_conv_embed=False, 
                use_layerscale=False, 
                layerscale_value=1e-4, 
                use_postln=False, 
                **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers)]

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        
        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size), 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim[0], 
            use_conv_embed=use_conv_embed, 
            norm_layer=norm_layer if self.patch_norm else None, 
            is_stem=True)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer], 
                               out_dim=embed_dim[i_layer+1] if (i_layer < self.num_layers - 1) else None,  
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate, 
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, 
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[i_layer], 
                               focal_window=focal_windows[i_layer], 
                               use_conv_embed=use_conv_embed,
                               use_layerscale=use_layerscale, 
                               layerscale_value=layerscale_value, 
                               use_postln=use_postln,
                    )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.dim_out = self.num_features

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x