import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

# 定义简单的卷积神经网络
#LicensePlateCNN 类继承自 nn.Module，定义了一个简单的卷积神经网络。
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        #网络包含两个卷积层（conv1 和 conv2）
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        #每个卷积层后跟着一个 ReLU 激活函数和一个最大池化层（pool1 和 pool2）。
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        #两个全连接层（fc1 和 fc2），用于将卷积层提取的特征映射到类别空间。
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    #forward 方法定义了数据在网络中的前向传播过程。
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(num):
    # 数据预处理
    #transforms.Compose 用于组合多个图像预处理操作。
    transform = transforms.Compose([
        transforms.Resize((224, 224)),#transforms.Resize 将图像调整为 224x224 大小。
        transforms.ToTensor(),#transforms.ToTensor 将图像转换为 PyTorch 张量。
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #transforms.Normalize 对图像进行归一化处理，使用预定义的均值和标准差。
    ])

    # 加载数据集
    #datasets.ImageFolder 用于从指定文件夹加载图像数据集，每个子文件夹代表一个类别。
    #DataLoader 用于将数据集封装成可迭代的数据加载器，方便批量处理数据。
    #shuffle=True 表示在每个训练周期开始时打乱训练数据的顺序，以增加模型的泛化能力。
    train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = datasets.ImageFolder(root='data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型、损失函数和优化器
    num_classes = len(train_dataset.classes)#num_classes 表示数据集中的类别数量。
    model = FaceCNN(num_classes)#LicensePlateCNN 实例化模型。
    criterion = nn.CrossEntropyLoss()#nn.CrossEntropyLoss 用于多分类问题的损失函数。
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optim.SGD 是随机梯度下降优化器，用于更新模型的参数。

    # 训练模型
    # #num_epochs 表示训练的轮数。
    num_epochs = num
    #device 用于指定使用 GPU 还是 CPU 进行训练。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device) 将模型移动到指定设备上。
    model.to(device)

    #每个训练周期中，遍历训练数据加载器，将图像和标签移动到指定设备上。
    #清零优化器的梯度，前向传播计算输出，计算损失，反向传播计算梯度，最后使用优化器更新模型参数。
    #打印每个训练周期的平均损失。
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    # 测试模型
    correct = 0
    total = 0
    #使用 torch.no_grad() 上下文管理器禁用梯度计算，以减少内存消耗。
    #遍历测试数据加载器，将图像和标签移动到指定设备上，前向传播计算输出。
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)#torch.max 函数用于获取预测结果的索引。
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')
    #统计正确预测的样本数量和总样本数量，计算并打印测试集的准确率。
    # 保存模型
    torch.save(model.state_dict(), 'face_model.pth')

    #字典映射
    class_names = train_dataset.classes
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)