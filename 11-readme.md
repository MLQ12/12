# 本文件将对接口类文件 11-classify.py 进行解释

# 导入库和模块

    torch：PyTorch库，用于深度学习。
    torchvision.models：包含了一些预定义的模型架构。
    dataset：一个自定义的模块，其中包含了CustomDataModule和otherDataset。
    numpy：用于数值计算。
    model：另一个自定义模块，其中包含了ViolenceClassifier类。
    torch.utils.data.DataLoader：用于加载和批处理数据。

# 定义ViolenceClass类

    定义了一个名为ViolenceClass的类，该类主要用于加载一个预训练的暴力分类模型，并对输入的图像进行分类。

## __init__函数

    接收两个参数：model_path（模型权重文件的路径）和use_gpu（一个布尔值，表示是否使用GPU进行推理）。
    使用ViolenceClassifier.load_from_checkpoint方法从给定的路径加载模型。
    将模型设置为评估模式（eval()）。
    如果use_gpu为True且GPU可用，则将模型移至GPU上；否则，模型将在CPU上运行。

## classify函数

    接收一个参数：imgs（一个形状为(n, 3, 224, 224)的Tensor，其中n是图像的数量，3是颜色通道数，224x224是图像的尺寸）。
    将图像移至模型所在的设备（CPU或GPU）。
    使用torch.no_grad()上下文管理器来确保在推理过程中不计算梯度（这可以节省内存和计算资源）。
    执行模型推理，得到输出。
    使用torch.max函数找到每个输出中的最大值（即最可能的类别）的索引。
    将预测结果从Tensor转换为Python列表。
    返回预测类别的列表。

# get_metrics函数

    该函数接受一个数据加载器 data 作为输入。
    初始化两个空列表 labels 和 preds，用于存储真实的标签和模型预测的标签。
    遍历数据加载器中的每一个批次（for i in data:），其中 i 是一个包含图像和对应标签的元组 (imgs, ys)。
    使用 classifier.classify(imgs) 方法对图像进行分类，并将预测结果存储在 pred 中。
    将真实的标签 ys 和预测的标签 pred 分别添加到 labels 和 preds 列表中。
    将 labels 和 preds 转换为 NumPy 数组，并确保它们是一维的。
    计算并返回分类准确率 acc。

# 主程序(__main__ 部分)

    这段代码主要是用来计算并输出在几个不同数据集上的模型分类准确率。
    首先，指定了模型的权重文件路径 model_path。
    使用 ViolenceClass 类加载模型，并创建一个 classifier 实例。
    创建一个 CustomDataModule 的实例 data_module，并设置其参数（这里假设 batch_size=132）。
    调用 data_module 的 setup 函数（尽管这里的结果 dd 没有被使用）。
    获取 data_module 的测试数据加载器 test1_dataloader。
    使用 otherDataset 加载两个额外的数据集，并分别创建数据加载器 test2_dataloader 和 test3_dataloader，这两个加载器的批次大小为 128 且不打乱数据。
    调用 get_metrics 函数，分别计算 test1_dataloader、test2_dataloader 和 test3_dataloader 上的模型分类准确率，并将结果存储在 test1_acc、test2_acc 和 test3_acc 中。
    打印出这三个数据集上的分类准确率。