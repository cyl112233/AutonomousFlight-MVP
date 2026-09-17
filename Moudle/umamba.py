import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class SSM_Block(nn.Module):
    def __init__(self, in_c, out_c, dp=0.1):
        super(SSM_Block, self).__init__()
        self.SSM = nn.Sequential(
        #     # nn.LayerNorm(in_c),
            Mamba(in_c, d_state=16, expand=2),
            nn.GELU(),
        #     # nn.Dropout(dp)
        )#引入状态空间模型
        self.Conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Dropout(dp),
        )#配合卷积操作

    def forward(self, x):
        # [Batch_size , Dim , H ,W] -> [Batch_size , Seq_len , Dim]
        # [Batch_size , Seq_len , Dim] Seq_len = H*W ：mamba需要接受的形状
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = self.SSM(x)

        x = x.reshape(B, H, W, C).transpose(1, 3)
        x = self.Conv(x)
        return x


class DownSampling(nn.Module):
    '''下采样模块'''
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变 ??
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        #x = self.Down(x)
        return self.Down(x)

class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍(是下面代码完成的)，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行上采样

        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


class UNet(nn.Module):

    def __init__(self, C_in, class_num):
        super(UNet, self).__init__()
        # 4次下采样
        self.Block = nn.Identity()
        self.C1 = SSM_Block(C_in, 64)
        self.D1 = DownSampling(64)
        self.C2 = SSM_Block(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = SSM_Block(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = SSM_Block(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = SSM_Block(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = SSM_Block(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = SSM_Block(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = SSM_Block(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = SSM_Block(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, class_num, 1, 1)

    def forward(self, x):

        # 下采样部分
        R1 = self.C1(x)
        # print("R1:", R1.shape)
        Block1 = self.Block(R1)
        # print("block1:", Block1.shape)
        R2 = self.C2(self.D1(R1))
        Block2 = self.Block(R2)

        R3 = self.C3(self.D2(R2))
        Block3 = self.Block(R3)

        R4 = self.C4(self.D3(R3))
        Block4 = self.Block(R4)

        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4)) + Block4
        O2 = self.C7(self.U2(O1, R3)) + Block3
        O3 = self.C8(self.U3(O2, R2)) + Block2
        # print("04:", self.C9(self.U4(O3, R1)).shape)
        O4 = self.C9(self.U4(O3, R1)) + Block1


        return self.Th(self.pred(O4))
# rand = torch.rand((1, 3, 256,256))
# MOUDLE = UNet(3, 2)
# # print(MOUDLE)
# OUTPUT = MOUDLE(rand)
# print(OUTPUT.shape)



# from PIL import Image
# import torchvision.transforms as trans
# trans_s = trans.Compose([trans.ToTensor(),
#                          trans.Resize(size=(256, 256))])
#
# Moudel = UNet(3, 3)
# image = r"D:\PythonProject\Vim-main\1723380554555.jpg"
# image = Image.open(image).convert('RGB')
# image = torch.unsqueeze(trans_s(image), 0)
# output = Moudel(image)

# import matplotlib.pyplot as plt
# import numpy as np
# output = torch.transpose(output, 1, 3).reshape(output.shape[2], output.shape[3],output.shape[1]).detach().numpy()
# print(output.shape)
# plt.imshow(output)
# plt.show()

