import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 数据集加载和预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testset = datasets.CIFAR10(root='./data', train=False,
                           download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)
# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 由于 CIFAR-10 图像是 32x32x3，我们先将其展平
        self.fc1 = nn.Linear(3 * 32 * 32, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    # 初始化模型、损失函数和优化器
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 主程序入口
if __name__ == '__main__':
    # 初始化模型、损失函数和优化器
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 训练模型并记录测试集错误率
num_epochs = 10
test_errors = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播 + 反向传播 + 优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个mini-batches打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            # 测试集上的评估
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_error = 100 - (100 * correct / total)
    test_errors.append(test_error)
    print('Epoch {}, Test Error: {:.2f}%'.format(epoch + 1, test_error))

# 绘制测试集错误率随 epoch 变化的图表
plt.plot(test_errors, marker='o')
plt.title('Test Error over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Test Error (%)')
plt.grid(True)
plt.show()  # 确保显示图表