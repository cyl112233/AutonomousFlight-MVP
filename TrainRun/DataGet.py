"""该文件的作用是用于数据集的加载"""
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy
Transform = transforms.Compose([
    transforms.ToTensor()
    ]
)

def re_size_Image(path,size=(256,256)):
    '''等比例缩放函数提去图片最大尺寸，然后构建一个为最大尺寸的mask掩码，在图片mask左上角处将图片复制'''
    image = Image.open(path).convert("RGB")
    max_liner = max(image.size)
    mask = Image.new("RGB",(max_liner,max_liner),(0,0,0))
    # 图片居中
    x = int((max_liner - image.size[0]) // 2)
    y = int((max_liner - image.size[1]) // 2)
    mask.paste(image,(x,y))
    mask = mask.resize(size)
    return mask

def re_size_Label(path,size=(256,256)):
    '''等比例缩放函数提去图片最大尺寸，然后构建一个为最大尺寸的mask掩码，在图片mask左上角处将图片复制'''
    image = Image.open(path).convert("L")

    max_liner = max(image.size)
    mask = Image.new("P",(max_liner,max_liner),(0,0,0))
    #图片居中
    x = int((max_liner - image.size[0]) // 2)
    y = int((max_liner - image.size[1]) // 2)
    mask.paste(image, (x, y))
    mask = mask.resize(size)

    return mask
class MyDateset(Dataset):
    '''
    root：根目录地址,
    image：根目录下图片名,
    lable：根目录下标签名
    size:图片尺寸
    ！！！注意标签名需要等于图片名，否则图片不会加载！！！
    '''
    def __init__(self,root,image,lable,size):
        '''此构造函数是构建图片和标签的列表'''

        self.size = size
        self.image_list = []
        self.label_list = []

        for i in os.listdir(str(os.path.join(root,lable))):
            self.image_set = os.path.join(root,image, i)
            self.image_list.append(self.image_set)
            self.label_set = os.path.join(root,lable, i)
            self.label_list.append(self.label_set)
    def __len__(self):
        return len(self.label_list)
    def __getitem__(self, indax):
        '''获取索引'''

        image, lable = self.image_list[indax], self.label_list[indax]
        '''缩放图片，默认是256*256'''
        image_re, lable_re = re_size_Image(image,size=self.size), re_size_Label(lable,size=self.size)
        lable_bn = numpy.array(lable_re,dtype=float)
        image_bn = Transform(image_re)


        return image_bn, lable_bn

        '''归一化'''
