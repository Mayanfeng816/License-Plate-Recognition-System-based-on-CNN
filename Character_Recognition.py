# 导入需要的库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 定义超参数
BATCH_SIZE = 64  # 每批训练数据的数量
NUM_EPOCHS = 100  # 训练轮数
LEARNING_RATE = 0.001  # 学习率

# 创建数据转换器
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),  # 将图像大小调整为32x32，字符切割直方图中字符大小是32
    transforms.Grayscale(),  # 转为灰度图像
    transforms.ToTensor(),  # 转为张量
])

# 加载数据集
train_dataset = datasets.ImageFolder('./train2/letter', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.ImageFolder('./test/letter', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 8 * 32, 128)
        self.fc2 = nn.Linear(128, 34)  # 共34个字符（26个字母+10个数字）

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# # 创建模型实例
# model = CNN()
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
# # 训练模型
# total_step = len(train_loader)
# for epoch in range(NUM_EPOCHS):
#     for i, (images, labels) in enumerate(train_loader):
#         # 向前传递
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # 每100个批次打印一次日志信息
#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))
#
# # 测试模型
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         #print(predicted)
#     print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
#
# # 保存模型
# torch.save(model.state_dict(), 'model/cnn_letter.ckpt')
