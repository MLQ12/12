import torch
import numpy as np
from model import ViolenceClassifier
import os
import random
from PIL import Image
from torchvision import transforms

def add_gaussian_noise(image_tensor, mean=0., std=0.1):
    """
    向图像张量添加高斯噪声
    :param image_tensor: 输入图像张量
    :param mean: 高斯分布的均值
    :param std: 高斯分布的标准差
    :return: 添加噪声后的图像张量
    """
    noise = torch.randn_like(image_tensor) * std + mean
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)  # 确保像素值在[0, 1]范围内
    return noisy_image

path = './violence_224/test'
data = os.listdir(path)
_0data = [i for i in data if i.split('_')[0] == '0']
_1data = [i for i in data if i.split('_')[0] == '1']

_0data = random.sample(_0data, 5)
_1data = random.sample(_1data, 5)

model_path = r'.\train_logs\resnet18_pretrain_test\version_7\checkpoints\resnet18_pretrain_test-epoch=24-val_loss=0.03.ckpt'
model = ViolenceClassifier.load_from_checkpoint(model_path)

for j in range(len([_0data, _1data])):
    d = [_0data, _1data][j]
    for i in d:
        img = os.path.join(path, i)
        img = Image.open(img)
        t = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor()
        ])
        img = t(img)
        img = add_gaussian_noise(img)
        
        toPIL = transforms.ToPILImage()
        img1 = toPIL(img)
        img1.save(f'./data/zaosheng/{i}')
