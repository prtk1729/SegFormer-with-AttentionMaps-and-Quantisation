import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from einops import rearrange
import segmentation_models_pytorch as smp
from timm.models.layers import drop_path, trunc_normal_



############################################################################
# 3 FUNDAMENTAL SUB-BLOCKS OF TRANSFORMER BLOCKS
############################################################################

################# SUB-BLOCK 1 #################
class overlap_patch_embed(nn.Module):
    '''
      - Can use nn.Conv2d as each patch will therby maintain 
      local continuity
      - Here, kernel_size is same as patch_size due to the simulation of 
      patches
      - Finally, each patch has to move to emb_dim-space as a vector 
      as shown above " Overlap Patch Embedding Idea in ViT image " after 
      the linear_projection (those unlabelled round rectangles)
      - einops is used to make the embs where there are (64 x 64) i.e
      4096 embs for each patch[do the math!] as a 64 dim-space
    '''
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size // 2, patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, h, w


################# SUB-BLOCK 2 #################
class efficient_self_attention(nn.Module):
    def __init__(self, attn_dim, num_heads, dropout_p, sr_ratio):
        super().__init__()
        assert attn_dim % num_heads == 0, f'expected attn_dim {attn_dim} to be a multiple of num_heads {num_heads}'
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(attn_dim, attn_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(attn_dim)

        # Multi-head Self-Attention using dot product
        # Query - Key Dot product is scaled by root of head_dim
        self.q = nn.Linear(attn_dim, attn_dim, bias=True)
        self.kv = nn.Linear(attn_dim, attn_dim * 2, bias=True)
        self.scale = (attn_dim // num_heads) ** -0.5

        # Projecting concatenated outputs from
        # multiple heads to single `attn_dim` size
        self.proj = nn.Linear(attn_dim, attn_dim)


    def forward(self, x, h, w):
        q = self.q(x)
        q = rearrange(q, ('b hw (m c) -> b m hw c'), m=self.num_heads)

        if self.sr_ratio > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.sr(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)

        x = self.kv(x)
        x = rearrange(x, 'b d (a m c) -> a b m d c', a=2, m=self.num_heads)
        k, v = x[0], x[1] # x.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = rearrange(x, 'b m hw c -> b hw (m c)')
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        attn_output = {'key' : k, 'query' : q, 'value' : v, 'attn' : attn}
        return x, attn_output
    


################# SUB-BLOCK 3 #################
class mix_feedforward(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, dropout_p = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Depth-wise separable convolution
        self.conv = nn.Conv2d(hidden_features, hidden_features, (3, 3), padding=(1, 1),
                              bias=True, groups=hidden_features)
        self.dropout_p = dropout_p

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class transformer_block(nn.Module):
    def __init__(self, dim, num_heads, dropout_p, drop_path_p, sr_ratio):
        super().__init__()
        # One transformer block is defined as :
        # Norm -> self-attention -> Norm -> FeedForward
        # skip-connections are added after attention and FF layers
        self.attn = efficient_self_attention(attn_dim=dim, num_heads=num_heads,
                    dropout_p=dropout_p, sr_ratio=sr_ratio)
        self.ffn = mix_feedforward( dim, dim, hidden_features=dim * 4, dropout_p=dropout_p)

        self.drop_path_p = drop_path_p
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)


    def forward(self, x, h, w):
        # Norm -> self-attention
        skip = x
        x = self.norm1(x)
        x, attn_output = self.attn(x, h, w)
        x = drop_path(x, drop_prob=self.drop_path_p, training=self.training)
        x = x + skip

        # Norm -> FeedForward
        skip = x
        x = self.norm2(x)
        x = self.ffn(x, h, w)
        x = drop_path(x, drop_prob=self.drop_path_p, training=self.training)
        x = x + skip
        return x, attn_output


class mix_transformer_stage(nn.Module):
    def __init__(self, patch_embed, blocks, norm):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm

    def forward(self, x):
        # patch embedding and store required data
        stage_output  = {}
        stage_output['patch_embed_input'] = x
        x, h, w = self.patch_embed(x)
        stage_output['patch_embed_h'] = h
        stage_output['patch_embed_w'] = w
        stage_output['patch_embed_output'] = x

        for block in self.blocks:
            x, attn_output = block(x, h, w)

        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        # store last attention block data
        # in stages' output data
        for k,v in attn_output.items():
            stage_output[k] = v
        del attn_output
        return x, stage_output


class Encoder(nn.Module):
    def __init__(self, in_chans, embed_dims, num_heads, depths,
                sr_ratios, dropout_p, drop_path_p):
        super().__init__()
        self.stages = nn.ModuleList()
        for stage_i in range(len(depths)):
            # Each Stage consists of following blocks :
            # Overlap patch embedding -> mix_transformer_block -> norm
            blocks = []
            for i in range(depths[stage_i]):
                blocks.append(transformer_block(dim = embed_dims[stage_i],
                        num_heads= num_heads[stage_i], dropout_p=dropout_p,
                        drop_path_p = drop_path_p * (sum(depths[:stage_i])+i) / (sum(depths)-1),
                        sr_ratio = sr_ratios[stage_i] ))
            # According to the stage_num we have diff patch size, stride 
            # and in_channels
            if(stage_i == 0):
                # according to the paper
                patch_size = 7
                stride = 4
                in_chans = in_chans # Since, this is first stage it interacts with image
            else:
                # according to the paper
                patch_size = 3
                stride = 2
                in_chans = embed_dims[stage_i -1] #recursive interaction with prev output embs

            patch_embed = overlap_patch_embed(patch_size, stride=stride, in_chans=in_chans,
                            embed_dim= embed_dims[stage_i])
            norm = nn.LayerNorm(embed_dims[stage_i], eps=1e-6)
            # mix transform is the smaller reactangle capsulating
            # Self-Attention and FFN in the diagram
            # stage is the mix_transform_stage -> entire 3 things
            # There are Nx of those in each stage
            # All are in sequential order => Put in a nn.ModuleList()
            # Total num of heads == Total num of such [OPM,ESA, FFN] == Total num of 
            self.stages.append(mix_transformer_stage(patch_embed, blocks, norm))



    def forward(self, x):
        outputs = []
        for i,stage in enumerate(self.stages):
            x, _ = stage(x)
            outputs.append(x)
        return outputs


    def get_attn_outputs(self, x):
        stage_outputs = []
        num_stages = 0
        for i,stage in enumerate(self.stages):
            num_stages += 1
            x, stage_data = stage(x)
            stage_outputs.append(stage_data)
        # print("==========>", num_stages)
        return stage_outputs



if __name__ == '__main__':
    # test here
    x = torch.randn(size=(8, 3, 512, 1024))
    in_channels = 3
    encoder = Encoder(in_chans=in_channels, embed_dims=(64, 128, 320, 512),
                                num_heads=(1, 2, 5, 8), depths=(3, 4, 18, 3),
                                sr_ratios=(8, 4, 2, 1), dropout_p=0.0, drop_path_p=0.1)
    enc_output = encoder(x)
    print("Transformer Stage 1 output shape: ", enc_output[0].shape) #torch.Size([8, 64, 128, 256])
    print("Transformer Stage 2 output shape: ", enc_output[1].shape) #torch.Size([8, 128, 64, 128])
    print("Transformer Stage 3 output shape: ", enc_output[2].shape) #torch.Size([8, 320, 32, 64])
    print("Transformer Stage 4 output shape: ", enc_output[3].shape) #torch.Size([8, 512, 16, 32])

    # This is sanity check as in Table 6 of the Paper, 
    # We can check the emb_dim in the 4 stages of Transformer Block
    # being -> (64, 128, 320, 512)
    # Other shapes also match
    # ==> Hence, Network is correct
    assert enc_output[0].shape == torch.Size([8, 64, 128, 256]) # sanity checks!! Tests
