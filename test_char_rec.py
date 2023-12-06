import torch
from sklearn import metrics
import matplotlib.pyplot as plt

test_dataset = datasets.ImageFolder('./test/letter', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 加载模型和测试数据
model = torch.load('model/cnn_letter.ckpt') # 假设模型已经保存为saved_model.pt文件
test_data = load_test_data() # 加载测试数据

# 使用模型进行预测
model.eval()
predicted_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted_labels += predicted.tolist()

# 计算模型的总体准确率、精确率、召回率和F1分数
accuracy = metrics.accuracy_score(test_data.targets, predicted_labels)
precision = metrics.precision_score(test_data.targets, predicted_labels, average='weighted')
recall = metrics.recall_score(test_data.targets, predicted_labels, average='weighted')
f1_score = metrics.f1_score(test_data.targets, predicted_labels, average='weighted')
print("准确率：", accuracy)
print("精确率：", precision)
print("召回率：", recall)
print("F1分数：", f1_score)

# 绘制评估结果的柱形图
x_labels = ['accuracy', 'precision', 'recall', 'f1_score']
scores = [accuracy, precision, recall, f1_score]
plt.bar(x_labels, scores)
plt.ylabel('Scores')
plt.title('Evaluation Results')
plt.show()
