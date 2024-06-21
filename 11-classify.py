import torch 
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  
subfolder_path = os.path.join(current_dir, '11-others')  
sys.path.append(subfolder_path)
from torchvision import models  
from dataset import CustomDataModule, otherDataset
import numpy as np
from model import ViolenceClassifier
from torch.utils.data import DataLoader

class ViolenceClass:  
    def __init__(self, model_path: str, use_gpu: bool = True):  
        """  
        初始化接口类，加载模型  
          
        :param model_path: 模型权重文件的路径  
        :param use_gpu: 是否使用GPU进行推理(默认为True)
        """  
        # 加载模型
        self.model = ViolenceClassifier.load_from_checkpoint(model_path)

  
        # 设置模型为评估模式  
        self.model.eval()  
  
        # 如果使用GPU，将模型移至GPU  
        if use_gpu and torch.cuda.is_available():  
            self.device = torch.device('cuda:0')  
            self.model = self.model.to(self.device)  
        else:  
            self.device = torch.device('cpu')  
  
    def classify(self, imgs: torch.Tensor) -> list:  
        """  
        对输入的图像进行分类  
          
        :param imgs: 输入的图像tensor,形状为(n, 3, 224, 224)  
        :return: 预测类别的列表,长度为n  
        """  
        imgs = imgs.to(self.device)  
  
        # 执行模型推理  
        with torch.no_grad():  # 无需计算梯度  
            outputs = self.model(imgs)  
  
            # 获取预测结果（假设使用softmax后的最大概率作为预测类别）  
            _, preds = torch.max(outputs, 1)  
  
            # 将tensor中的预测结果转换为Python列表  
            preds_list = preds.detach().cpu().numpy().tolist()
        return preds_list  
  
# 使用示例  
# 假设已经有一个预训练好的模型权重文件'model_weights.pth'  
# 并且有一个CustomDataModule的实例（例如data_module）来加载数据  

def get_metrics(data):
    # 你可以遍历test_dataloader来获取图像并进行分类
    labels, preds = [], []  
    for i in data:  
        imgs, ys = i
        pred = classifier.classify(imgs)

        labels.extend(ys.numpy().tolist())
        preds.extend(pred)

    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)
    acc = sum([labels[i] == preds[i] for i in range(len(labels))]) / len(labels)
    return acc

if __name__ == '__main__':
    # 加载模型  
    model_path = r'.\11-others\train_logs\resnet18_pretrain_test\version_7\checkpoints\resnet18_pretrain_test-epoch=24-val_loss=0.03.ckpt'
    classifier = ViolenceClass(model_path=model_path)  
    
    # 假设你有一个CustomDataModule的实例，你可以使用它来加载测试数据  
    data_module = CustomDataModule(batch_size=132)  
    dd = data_module.setup()
    test1_dataloader = data_module.test_dataloader()  
    test2_dataloader = otherDataset('./11-others/data/aigc')
    test2_dataloader = DataLoader(test2_dataloader, batch_size=128, shuffle=False)
    test3_dataloader = otherDataset('./11-others/data/zaosheng')
    test3_dataloader = DataLoader(test3_dataloader, batch_size=128, shuffle=False)
    

    test1_acc = get_metrics(test1_dataloader)
    test2_acc = get_metrics(test2_dataloader)
    test3_acc = get_metrics(test3_dataloader)

    print(f'test1 acc:{test1_acc}, test2 acc:{test2_acc}, test3 acc:{test3_acc}')