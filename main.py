import random
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from efficientnet import EfficientNetv2, efficientnetv2_l, efficientnetv2_s
from net import vgg16
from torch.utils.data import DataLoader
from data import *
from resnet import ResNet, ResNet18, ResNet50
from shufflenet import ShuffleNet

'''数据集'''
annotation_path = 'new_cls.txt'
with open(annotation_path, 'r') as f:
    lines = f.readlines()
np.random.seed(random.randint(0, 123456))
np.random.shuffle(lines)  # 打乱数据
np.random.seed(None)
num_val = int(len(lines) * 0.1)
num_train = len(lines) - num_val
# 输入图像大小
input_shape = [224, 224]
train_data = DataGenerator(lines[:num_train], input_shape, True)
val_data = DataGenerator(lines[num_train:], input_shape, False)
val_len = len(val_data)
"""加载数据"""
gen_train = DataLoader(train_data, batch_size=16, shuffle=True)
gen_test = DataLoader(val_data, batch_size=16, shuffle=True)
'''构建网络'''
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
net = ResNet18(in_channels=3,num_classes=2).to(device)
'''选择优化器和学习率的调整方法'''
lr = 0.001
optim = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)
total_test = 0
total_accuracy = 0
'''训练'''
epochs = 200
for epoch in range(epochs):
    # if epoch % 30 == 0:
    #     lr = lr * 0.1
    #     optim = torch.optim.Adam(net.parameters(), lr=lr)
    total_train = 0
    net.train()
    for data in gen_train:
        img, label = data
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
        optim.zero_grad()
        output = net(img)
        train_loss = nn.CrossEntropyLoss()(output, label).to(device)
        train_loss.backward()
        optim.step()
        total_train += train_loss
    scheduler.step()
    total_test = 0
    total_accuracy = 0
    net.eval()
    for data in gen_test:
        img, label = data
        print(label)
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            optim.zero_grad()
            out = net(img)
            score = out[:,1]
            auc = roc_auc_score(label,score)
            test_loss = nn.CrossEntropyLoss()(out, label).to(device)
            total_test += test_loss
            accuracy = (out.argmax(1) == label).sum()
            for i in range(0, 9):
                if out.argmax(1)[i] != label[i]:
                    print(label[i])
            total_accuracy += accuracy
    # torch.save(net.state_dict(), "hasTLS.{}.pth".format(epoch + 1))
    print("epoch: {}, train_loss: {},  test_loss: {},  auc: {}  , acc: {}, lr: {}".format(epoch, total_train, total_test,
                                                                                   total_accuracy / val_len,auc,
                                                                                   optim.param_groups[0]['lr']))
print("test_loss: {}, accuracy: {}".format(total_test, total_accuracy / val_len))
