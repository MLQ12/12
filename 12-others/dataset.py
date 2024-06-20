from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, split):
        assert split in ["train", "val", "test"]
        data_root = r".\violence_224"
        split_dir = os.path.join(data_root, split)
        self.data = [os.path.join(split_dir, i) for i in os.listdir(split_dir)]  
        if not self.data:  
            raise ValueError(f"The {split} dataset is empty. Please check the {split_dir} directory.")    

        if split == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 随机翻转
                transforms.ToTensor(),  # 将图像转换为Tensor
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),  # 将图像转换为Tensor
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  
        img_path = self.data[index]  
        # 使用os.path.basename获取文件名，并使用splitext分割文件名和扩展名  
        filename_with_ext = os.path.basename(img_path)  
        filename, file_extension = os.path.splitext(filename_with_ext)  
        # 获取标签值，0代表非暴力，1代表暴力  
        # 假设文件名的主体（不包括扩展名）的第一个字符是标签  
        y = int(filename[0])  
  
        # 验证标签是否为0或1  
        if y not in [0, 1]:  
            raise ValueError(f"Invalid label: {y} in file {img_path}")  
  
        x = Image.open(img_path)  
        x = self.transforms(x)  
        return x, y  


class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 分割数据集、应用变换等
        # 创建 training, validation数据集
        self.train_dataset = CustomDataset("train")
        self.val_dataset = CustomDataset("val")
        self.test_dataset = CustomDataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class otherDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.data = [os.path.join(path, i) for i in os.listdir(path)]

        self.t = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = self.data[index]  
        # 使用os.path.basename获取文件名，并使用splitext分割文件名和扩展名  
        filename_with_ext = os.path.basename(img_path)  
        filename, file_extension = os.path.splitext(filename_with_ext)  
        # 获取标签值，0代表非暴力，1代表暴力  
        # 假设文件名的主体（不包括扩展名）的第一个字符是标签  
        y = int(filename[0])  
  
        # 验证标签是否为0或1  
        if y not in [0, 1]:  
            raise ValueError(f"Invalid label: {y} in file {img_path}")  
  
        x = Image.open(img_path)  
        x = self.t(x)  
        return x, y 

    def __len__(self):
        return len(self.data) 

import torch
if __name__ == '__main__':
    # data = CustomDataModule(1)
    # dd = data.setup()
    # train = data.train_dataloader()
    # for i in train:
    #     x, y = i
    #     print(x)
    #     print(torch.max(x), torch.min(x))
    #     print(y)
    #     break
    get_data('./data/aigc')