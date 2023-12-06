import Character_Predict_zh as CPZH
import Character_Predict as CPLT


def predict(path):
    dict_ccpd_zh = {
        '0': '皖',
        '1': '沪',
        '2': '津',
        '3': '渝',
        '4': '冀',
        '5': '晋',
        '6': '蒙',
        '7': '辽',
        '8': '吉',
        '9': '黑',
        '10': '苏',
        '11': '浙',
        '12': '京',
        '13': '闽',
        '14': '赣',
        '15': '鲁',
        '16': '豫',
        '17': '鄂',
        '18': '湘',
        '19': '粤',
        '20': '桂',
        '21': '琼',
        '22': '川',
        '23': '贵',
        '24': '云',
        '25': '藏',
        '26': '陕',
        '27': '甘',
        '28': '青',
        '29': '宁',
        '30': '新',
    }

    dict_ccpd_letter = {
        "0": 'A',
        "1": 'B',
        "2": 'C',
        "3": 'D',
        '4': 'E',
        '5': 'F',
        '6': 'G',
        '7': 'H',
        '8': 'J',
        '9': 'K',
        '10': 'L',
        '11': 'M',
        '12': 'N',
        '13': 'P',
        '14': 'Q',
        '15': 'R',
        '16': 'S',
        '17': 'T',
        '18': 'U',
        '19': 'V',
        '20': 'W',
        '21': 'X',
        '22': 'Y',
        '23': 'Z',
        '24': '0',
        '25': '1',
        '26': '2',
        '27': '3',
        '28': '4',
        '29': '5',
        '30': '6',
        '31': '7',
        '32': '8',
        '33': '9',
    }
    zh = CPZH.CPZH(path)
    letter = CPLT.CPLT(path)
    result = zh + letter
    print("预测结果：" + result)

    string = path
    label = string.split("_", 1)
    temp = list(label[1].split("_"))
    label_zh = dict_ccpd_zh[label[0]]
    label_letter = ""
    for item in temp:
        label_letter += dict_ccpd_letter[item]
    car_label = label_zh + label_letter
    print("车牌实际：" + car_label)

    correct = 0
    for i in range(7):
        if result[i] == car_label[i]:
            correct = correct + 1
    acc = correct / 7
    print('正确个数：{0}'.format(correct))
    print('准确率：%.2f%%' % (acc * 100))

    return result,acc