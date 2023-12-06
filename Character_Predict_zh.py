import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from Character_Recognition import CNN
def CPZH(path):
    dict1={'0':'川',
           '1':'鄂',
           '2':'赣',
           '3':'甘',
           '4':'贵',
           '5':'桂',
           '6':'黑',
           '7':'沪',
           '8':'冀',
           '9':'津',
           '10':'京',
           '11':'吉',
           '12':'辽',
           '13':'鲁',
           '14':'蒙',
           '15':'闽',
           '16':'宁',
           '17':'青',
           '18':'琼',
           '19':'陕',
           '20':'苏',
           '21':'晋',
           '22':'皖',
           '23':'湘',
           '24':'新',
           '25':'豫',
           '26':'渝',
           '27':'粤',
           '28':'云',
           '29':'藏',
           '30':'浙',
           }

    #模型的加载
    # save_path，和模型的保存那里的save_path一样
    # .eval() 预测结果前必须要做的步骤，其作用为将模型转为evaluation模式
    # Sets the module in evaluation mode.
    model = CNN()
    model.load_state_dict(torch.load("./model/cnn_zh.ckpt"))
    model.eval()

    # 定义超参数
    BATCH_SIZE = 1  # 每批训练数据的数量
    # 创建数据转换器
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),  # 将图像大小调整为32x32
        transforms.Grayscale(),  # 转为灰度图像
        transforms.ToTensor(),  # 转为张量
    ])

    test_dataset = datasets.ImageFolder('./ture_car_characters/'+path+'/zh', transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            car_license_zh=""
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            value=predicted.tolist()
            value_str = list(map(str, value))
            print(value_str)
            for item in value_str:
                car_license_zh+=dict1[item]
            print(car_license_zh)
            return car_license_zh
