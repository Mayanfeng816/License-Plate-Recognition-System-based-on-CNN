import pickle

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os

# 定义训练数据文件夹路径
train_data_path = 'svm/train/'

# 定义SVM模型超参数
svm_params = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['rbf']
}

# 定义图像大小
IMG_WIDTH = 350
IMG_HEIGHT = 120
#
#
# 读取图像和标签
def prepare_data(train_folder_path):
    train_data = []
    train_labels = []

    for label, folder_name in enumerate(os.listdir(train_folder_path)):
        folder_path = os.path.join(train_folder_path, folder_name)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, 0)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = np.array(image).flatten()
            train_data.append(image)
            train_labels.append(label)
    return train_data, train_labels

model_filename = 'svm_model.pkl'

# # 准备训练数据
train_data, train_labels = prepare_data(train_data_path)

# 训练SVM模型
svm_model = SVC()
grid_search = GridSearchCV(svm_model, svm_params)
grid_search.fit(
    train_data, train_labels)

# 保存SVM模型

with open(model_filename, 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

#加载SVM模型
# with open(model_filename, 'rb') as f:
#     svm_model = pickle.load(f)


#加载测试数据，进行预测
# def predict(test_data):
#     pred = svm_model.predict(test_data)
#     #获取距离超平面距离，0是所求车牌图片，距离越小，概率越大
#     dest = svm_model.decision_function(test_data)
#     print(pred)
#     print(dest)
#     print(np.argmin(np.array(dest)))
#     return np.argmin(np.array(dest))

# 加载模型并测试
# test_data=[]
# for j in os.listdir('temp-roi'):
#     test_path = os.path.join('temp-roi',j)
#     image = cv2.imread(test_path, 0)
#     image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
#     image = np.array(image).flatten()
#     test_data.append(image)
# predict(test_data)



