import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from Character_Recognition import CNN

def CPLT(path):
    dict1={"0":'A',
           "1":'B',
           "2":'C',
           "3":'D',
           '4':'E',
           '5':'F',
           '6':'G',
           '7':'H',
           '8':'J',
           '9':'K',
           '10':'L',
           '11':'M',
           '12':'N',
           '13':'P',
           '14':'Q',
           '15':'R',
           '16':'S',
           '17':'T',
           '18':'U',
           '19':'V',
           '20':'W',
           '21':'X',
           '22':'Y',
           '23':'Z',
           '24':'0',
           '25':'1',
           '26':'2',
           '27':'3',
           '28':'4',
           '29':'5',
           '30':'6',
           '31':'7',
           '32':'8',
           '33':'9',
           '34':'川',
           '35':'鄂',
           '36':'赣',
           '37':'甘',
           '38':'贵',
           '39':'桂',
           '40':'黑',
           '41':'沪',
           '42':'冀',
           '43':'津',
           '44':'京',
           '45':'吉',
           '46':'辽',
           '47':'鲁',
           '48':'蒙',
           '49':'闽',
           '50':'宁',
           '51':'青',
           '52':'琼',
           '53':'陕',
           '54':'苏',
           '55':'晋',
           '56':'皖',
           '57':'湘',
           '58':'新',
           '59':'豫',
           '60':'渝',
           '61':'粤',
           '62':'云',
           '63':'藏',
           '64':'浙',
           }

    #模型的加载
    # save_path，和模型的保存那里的save_path一样
    # .eval() 预测结果前必须要做的步骤，其作用为将模型转为evaluation模式
    # Sets the module in evaluation mode.
    model = CNN()
    model.load_state_dict(torch.load("./model/cnn_letter.ckpt"))
    model.eval()

    # 定义超参数
    BATCH_SIZE = 6  # 每批训练数据的数量
    # 创建数据转换器
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),  # 将图像大小调整为32x32
        transforms.Grayscale(),  # 转为灰度图像
        transforms.ToTensor(),  # 转为张量
    ])

    test_dataset = datasets.ImageFolder('./ture_car_characters/'+path+'/letter', transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            car_license_letter=""
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            value=predicted.tolist()
            value_str = list(map(str, value))
            print(value_str)
            for item in value_str:
                car_license_letter+=dict1[item]
            print(car_license_letter)
            return car_license_letter
