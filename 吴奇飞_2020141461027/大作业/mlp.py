import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import time
from loguru import logger

import torchvision
import torchvision.transforms as transforms

# 配置loguru记录到文件

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ComplexMLP(nn.Module):
    def __init__(self, input_size):
        super(ComplexMLP, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, self.input_size)  # 展平成一维向量
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc5(x)
        return x

def evaluate(net, testloader, device):
    correct = 0
    total = 0
    net.eval()  # 设置模型为评估模式
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.warning(f'[eval]Accuracy of the network on the {len(testloader.dataset)} test images: {accuracy:.2f}%')
    net.train()  # 评估后重新设置模型为训练模式

def train(net, trainloader,testloader, device, optimizer, criterion, scheduler,epoch_num):
    for epoch in range(epoch_num):  # 训练50个epoch
        a = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                logger.info("epoch {}[batch i:{}][batch size:{}][loss:{:.2f}]".format(epoch+1, i+1, batch_size, running_loss/100))
                running_loss = 0.0
        scheduler.step()
        b = time.time()
        logger.info("epoch {} time:{:.2f}s".format(epoch+1, b - a))
        evaluate(net, testloader, device)

if __name__ == '__main__':
    input_size=3*32*32
    batch_size = 64
    epoch_num=50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.add("mlp_training.log", level="INFO")
    logger.warning("device.type:{}".format(device.type))
    logger.warning("batch_size:{}".format(batch_size))
    logger.warning("epoch_num:{}".format(epoch_num))

    # 数据预处理
    #对于 CIFAR-10 数据集，常见的标准化参数如下：
    #均值(0.4914, 0.4822, 0.4465)
    #标准差(0.2470, 0.2435, 0.2616)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    # 加载CIFAR-10训练集
    trainset = torchvision.datasets.CIFAR10(root=r"E:\dataset\cifar10", train=True, download=False, transform=transform)

    subset_indices = list(range(min(len(trainset), 20000)))  # 选择前n个样本
    subset_trainset = Subset(trainset, subset_indices)
    trainloader = DataLoader(subset_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    # 加载CIFAR-10测试集
    testset = torchvision.datasets.CIFAR10(root=r"E:\dataset\cifar10", train=False, download=False, transform=transform)
    subset_indices = list(range(min(len(testset), 10000)))  # 选择前n个样本
    subset_testset = Subset(testset, subset_indices)
    testloader = DataLoader(subset_testset, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info("trainset size:{}".format(len(subset_trainset)))
    logger.info("testset size:{}".format(len(subset_testset)))

    net = ComplexMLP(input_size)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train(net, trainloader,testloader, device, optimizer, criterion, scheduler,epoch_num)
    logger.info('Finished Training')
    torch.save(net,"mlp.pth")
