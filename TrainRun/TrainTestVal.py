import torch
from TrainRun.Label2Image import label_to_RGB,segmentation_example
from torchvision.utils import save_image
from TrainRun.eval_function import SegmentationMetric
import numpy as np
import cv2
class train_test_val():
    def __init__(self,class_num,save_epoch):
        self.class_num = class_num
        self.save_epoch = save_epoch

    def train(self,i,train_data_loader,Cuda_num,Moudle,loss_function,Optimizer,learning,label_colors,look=False):
            Moudle.train()
            loss_sum = 0
            mIoU_sum = 0
            pa_sum = 0
            for input, lable in train_data_loader:

                #设备转换 Host_to_Device
                input, lable = input.cuda(device=Cuda_num[0]), lable.cuda(device=Cuda_num[0])

                output = Moudle(input)
                loss = loss_function(output, lable.long())

                #评估计算
                output = torch.argmax(output, dim=1)

                metric = SegmentationMetric(self.class_num)  # 分类数
                metric.addBatch(output.cpu(), lable.cpu())  # 预测图片，标签图片
                pa = metric.pixelAccuracy()
                mIoU = metric.meanIntersectionOverUnion()
                loss_sum += loss.item()
                pa_sum +=pa
                mIoU_sum += mIoU


                # 梯度清零,反向传播，返向传播，优化器,学习率递减
                Optimizer.zero_grad()
                loss.backward()
                Optimizer.step()
                learning.step()
                # cpa = metric.classPixelAccuracy()
                # mpa = metric.meanPixelAccuracy()
                #可视化训练过程
                if i % self.save_epoch == 0 and look == True:
                    #设备转移
                    p_image = input[0].cpu()
                    p_lable_p = lable[0].cpu()
                    p_output_p = output[0].cpu()

                    #Label image转换RGB image
                    p_lable, p_output = (label_to_RGB(p_lable_p, label_colors),
                                         label_to_RGB(p_output_p,label_colors)
                                         )
                    #分割图像保存
                    int_p_image = p_image*255
                    segmentation_image = segmentation_example(np.transpose(np.array(int_p_image,dtype=np.uint8)),
                                                              np.transpose(np.array(p_output)))
                    cv2.imwrite(f"./SaveDate/ImageDate/TestSeg{i}.png", segmentation_image)
                    # 统一为tensor格式
                    p_lable, p_output = torch.tensor(p_lable), torch.tensor(p_output)
                    s_image = torch.stack([p_image.float(),p_lable.float(), p_output.float()], dim=0)
                    #保存
                    save_image(s_image, f"./SaveDate/ImageDate/Train{i}.png")

            return loss_sum,pa_sum,mIoU_sum

    def test(self,i,test_data_loader,Cuda_num,Moudle,loss_function,label_colors,look=False):
            Moudle.eval()
            with (torch.no_grad()):
                loss_sum = 0
                mIoU_sum = 0
                pa_sum = 0
                for input, lable in test_data_loader:

                    # 设备转换 Host_to_Device
                    input, lable = input.cuda(device=Cuda_num[0]), lable.cuda(device=Cuda_num[0])

                    output = Moudle(input)
                    loss = loss_function(output, lable.long())

                    # 评估计算
                    output = torch.argmax(output, dim=1)

                    metric = SegmentationMetric(self.class_num)  # 分类数
                    metric.addBatch(output.cpu(), lable.cpu())  # 预测图片，标签图片
                    pa = metric.pixelAccuracy()
                    mIoU = metric.meanIntersectionOverUnion()
                    loss_sum += loss.item()
                    pa_sum += pa
                    mIoU_sum += mIoU

                    # 可视化训练过程
                    if i % self.save_epoch == 0 and look == True:
                        # 设备转移
                        p_image = input[0].cpu()
                        p_lable_p = lable[0].cpu()
                        p_output_p = output[0].cpu()
                        # Label image转换RGB image
                        p_lable, p_output = (label_to_RGB(p_lable_p, label_colors),
                                             label_to_RGB(p_output_p, label_colors)
                                             )
                        # 分割图像保存
                        int_p_image = p_image * 255
                        segmentation_image = segmentation_example(np.transpose(np.array(int_p_image, dtype=np.uint8)),
                                                                  np.transpose(np.array(p_output)))
                        cv2.imwrite(f"./SaveDate/ImageDate/TestSeg{i}.png", segmentation_image)
                        #统一为tensor格式
                        p_lable, p_output = torch.tensor(p_lable), torch.tensor(p_output)
                        s_image = torch.stack([p_image.float(), p_lable.float(), p_output.float()], dim=0)
                        #保存
                        save_image(s_image, f"./SaveDate/ImageDate/Test{i}.png")
                return loss_sum, pa_sum, mIoU_sum