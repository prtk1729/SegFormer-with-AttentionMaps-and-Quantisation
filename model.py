from encoder import Encoder
from decoder import Decoder
import torch
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import torch.nn as nn

class segformer_mit_b3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Encoder block
        self.backbone = Encoder(in_chans=in_channels, embed_dims=(64, 128, 320, 512),
                                    num_heads=(1, 2, 5, 8), depths=(3, 4, 18, 3),
                                    sr_ratios=(8, 4, 2, 1), dropout_p=0.0, drop_path_p=0.1)
        # decoder block
        self.decoder_head = Decoder(in_channels=(64, 128, 320, 512),
                                    num_classes=num_classes, embed_dim=256)

        # init weights
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


    def forward(self, x):
        image_hw = x.shape[2:]
        x = self.backbone(x)
        x = self.decoder_head(x)
        x = F.interpolate(x, size=image_hw, mode='bilinear', align_corners=False)
        return x


    def get_attention_outputs(self, x):
        return self.backbone.get_attn_outputs(x)

    def get_last_selfattention(self, x):
        outputs = self.get_attention_outputs(x)        
        return outputs[-1].get('attn', None)

    def get_selfattention_for_any_stage(self, x, stage_num):
      outputs = self.get_attention_outputs(x)
      for i in range(len(outputs)):
        print("Shape: ", outputs[i].get('attn', None).shape)      
      return outputs[stage_num].get('attn', None)


if __name__ == '__main__':
    # test here
    x = torch.randn(size=(8, 3, 512, 1024))
    in_channels = 3 # RGB image
    num_classes = 19 # for cityscapes

    model = segformer_mit_b3(in_channels=in_channels, num_classes=num_classes)
    out = model(x)
    print( out.shape ) # torch.Size([8, 19, 512, 1024])

    # Pick an any image in a batch
    # [512, 1024] -> [h, w]
    # Now for every pixel -> we have 19 prob values
    # If we apply argmax on each pixel we get its predicted class