import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
class SegMoudle(nn.Module):
    def __init__(self,num_classes,embed_dim=768,freeze_backbone=True):
        super().__init__()
        self.Mode = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).cuda()

        self.Mode.heads = nn.Identity()


        # 获取 ViT 的隐藏维度（通常是 768）
        self.hidden_dim = self.Mode.hidden_dim  # 768
        self.patch_size = self.Mode.patch_size  # 16
        # 可选：冻结 backbone 参数
        if freeze_backbone:
            for param in self.Mode.parameters():
                param.requires_grad = False
        self.cnn = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.mamba = nn.Sequential(Mamba(embed_dim, d_state=16, expand=2),
                                   nn.LayerNorm(embed_dim),
                                   nn.ReLU(inplace=True),
                                   )
        self.mamba2 = nn.Sequential(Mamba(embed_dim*2, d_state=16, expand=2),
                                   nn.LayerNorm(embed_dim),
                                   nn.ReLU(inplace=True),
                                   )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )


    def forward(self, x):
        B, C, H, W = x.shape
        x = self.Mode.conv_proj(x)  # (B, hidden_dim, H/p, W/p)
        cnn = self.cnn(x)
        mamba = self.mamba(cnn.flatten(2).transpose(1, 2))
        Mixed_features = torch.cat((x.flatten(2).transpose(1, 2), mamba), dim=2)

        mamba2 = self.mamba2(Mixed_features)
        x = self.mlp(mamba2)


        cls_token = self.Mode.class_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        # print(cls_token.shape)
        x = torch.cat([cls_token, x], dim=1)  # (B, N+1, hidden_dim)

        x = self.Mode.encoder(x)  # (B, N+1, hidden_dim)

        # 4. 去掉 CLS token，只保留 patch tokens
        patch_tokens = x[:, 1:, :]


        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        D = patch_tokens.shape[-1]
        feat_map = patch_tokens.permute(0, 2, 1).contiguous().view(B, D, grid_h, grid_w)

        # 6. 通过分割头
        out = self.seg_head(feat_map)  # (B, num_classes, H', W')

        # 如果输出尺寸与输入不一致，用双线性插值调整
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out
# data = torch.randn(1,3,224,224).cuda()
# moudle = SegMoudle(3).cuda()
# # print(moudle)
# moudle.eval()
# print(moudle(data).shape)
