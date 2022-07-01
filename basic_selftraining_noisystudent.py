# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 00:01:57 2022

@author: WooYoungHwang
"""
import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_util
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Load Cifar 10 data
# =============================================================================
TRAIN_DATADIR = r'C:\Users\WooYoungHwang\Desktop\SPS\데이터\CIFAR-10-images-master\CIFAR-10-images-master\train'
CATEGORIES = ['airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# show image
for category in CATEGORIES:
    path = os.path.join(TRAIN_DATADIR, category)
    for img in os.listdir(path):
        img_array = np.fromfile(os.path.join(path, img), np.uint8)
        curImg = cv2.imdecode(img_array,  cv2.IMREAD_COLOR)
        plt.imshow(curImg)
        plt.show()
        break
    break
curImg.shape
IMG_SIZE = curImg.shape[0]

# =============================================================================
# create train data
# =============================================================================
def create_training_data(slice_num_start, slice_num_end, data_list):
    for i in range(len(CATEGORIES)):
        category = CATEGORIES[i]
        path = os.path.join(TRAIN_DATADIR, category)
        for img in os.listdir(path)[slice_num_start:slice_num_end]:
            img_array = np.fromfile(os.path.join(path, img), np.uint8)
            curImg = cv2.imdecode(img_array,  cv2.IMREAD_COLOR)
            data_list.append([curImg, i])

    return data_list

# load train data       
training_data = []     
training_dataset = create_training_data(0, 1000, training_data)

# teacher model train data
train_dataloader = data_util.DataLoader(training_dataset, batch_size=64, shuffle=True)


# =============================================================================
# create psuedo data
# =============================================================================
# load psuedo data
psuedo_data = []
psuedo_dataset = create_training_data(1000, 1500, psuedo_data)
psuedo_dataset = random.sample(psuedo_dataset, len(psuedo_dataset))


# =============================================================================
# create test data
# =============================================================================
TEST_DATADIR = r'C:\Users\WooYoungHwang\Desktop\SPS\데이터\CIFAR-10-images-master\CIFAR-10-images-master\test'

def create_test_data(slice_num_start, slice_num_end, data_list):
    for i in range(len(CATEGORIES)):
        category = CATEGORIES[i]
        path = os.path.join(TEST_DATADIR, category)
        for img in os.listdir(path)[slice_num_start:slice_num_end]:
            img_array = np.fromfile(os.path.join(path, img), np.uint8)
            curImg = cv2.imdecode(img_array,  cv2.IMREAD_COLOR)
            data_list.append([curImg, i])

    return data_list

# load test data
test_dataset = []
test_dataset = create_test_data(0, 300, test_dataset)
test_dataset = random.sample(test_dataset, len(test_dataset))


# =============================================================================
# Baseline Model
# =============================================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1176, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =============================================================================
# Train Teacher Model
# =============================================================================
#initialize teacher model
teacher = Net().to(device)
# model parameter
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(teacher.parameters(), lr=0.0001)
epoch = 100

loss_history = []
## train
for epoch in tqdm(range(epoch)):   # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs = torch.tensor(inputs.permute(0,3,1,2)).float().to(device)
        labels = labels.to(device)
        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = teacher(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 통계를 출력합니다.
        running_loss += loss.item()
        print('%s=> loss: %.3f' %(i, loss))
        loss_history.append(loss.cpu().detach())

print('Finished Training')

#plot training loss
plt.plot(np.array(loss_history))

# # Save Model
save_module_PATH = r'C:\Users\WooYoungHwang\Desktop\SPS\2022 spring semester\다변량통계분석및데이터마이닝\과제\과제7\코드'           #모듈 저장 위치 지정
torch.save(teacher.state_dict(), save_module_PATH + 'cifar10_teacher.pt')

# # Load Model
load_module_PATH = r'C:\Users\WooYoungHwang\Desktop\SPS\2022 spring semester\다변량통계분석및데이터마이닝\과제\과제7\코드'  
teacher = Net()
teacher.load_state_dict(torch.load(load_module_PATH + 'cifar10_teacher.pt'))
teacher = teacher.to(device)


# =============================================================================
# Psuedo Labeling
# =============================================================================
new_psuedo_dataset = []
for i, data in enumerate(psuedo_dataset, 0):
    # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
    inputs, labels = data
    inputs = torch.tensor(inputs).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)

    # create pseudo label
    outputs = teacher(inputs)
    new_label = torch.softmax(outputs, -1)
    y = int(torch.argmax(new_label, dim=1))
    new_psuedo_dataset.append([np.array(inputs.squeeze(0).permute(1, 2, 0).cpu().detach().int()), y])

# create psuedo data (append train data with pseudo data)
student_dataset = training_dataset + new_psuedo_dataset
new_psuedo_dataloader = data_util.DataLoader(student_dataset, batch_size=64, shuffle=True)


# =============================================================================
# Train Student Model
# =============================================================================
loss_history = []
## train
for epoch in tqdm(range(epoch)):   # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(new_psuedo_dataloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs = torch.tensor(inputs.permute(0,3,1,2)).float().to(device)
        labels = labels.to(device)
        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = teacher(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 통계를 출력합니다.
        running_loss += loss.item()
        print('%s=> loss: %.3f' %(i, loss))
        loss_history.append(loss.cpu().detach())

print('Finished Training')
plt.plot(np.array(loss_history))

# # Save Model
save_module_PATH = r'C:\Users\WooYoungHwang\Desktop\SPS\2022 spring semester\다변량통계분석및데이터마이닝\과제\과제7\코드'           #모듈 저장 위치 지정
torch.save(teacher.state_dict(), save_module_PATH + 'cifar10_student.pt')

# # Load Model
load_module_PATH = r'C:\Users\WooYoungHwang\Desktop\SPS\2022 spring semester\다변량통계분석및데이터마이닝\과제\과제7\코드'  
student = Net()
student.load_state_dict(torch.load(load_module_PATH + 'cifar10_student.pt'))
student = student.to(device)


# =============================================================================
# Test Self-Training Data
# =============================================================================
y_pred = []
y_true = []
for i, data in enumerate(test_dataset, 0):
    # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
    inputs, labels = data
    inputs = torch.tensor(inputs).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)

    # create pseudo label
    outputs = student(inputs)
    new_label = torch.softmax(outputs, -1)
    y = int(torch.argmax(new_label, dim=1))
    y_pred.append(y)
    y_true.append(labels)

print(classification_report(y_true, y_pred, target_names=CATEGORIES))


# =============================================================================
# Create Noisy Student dataset
# =============================================================================
student_dataset = training_dataset + new_psuedo_dataset
noisystudent_dataloader = data_util.DataLoader(student_dataset, batch_size=64, shuffle=True)

# =============================================================================
# Noisy Student Model
# =============================================================================
class NoisyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.conv3 = nn.Conv2d(16, 32, 2)
        self.fc1 = nn.Linear(288, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# =============================================================================
# Train Teacher Model
# =============================================================================
#initialize teacher model
student_model = NoisyStudent().to(device)
# model parameter
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
epoch = 100

loss_history = []
## train
for epoch in tqdm(range(epoch)):   # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(noisystudent_dataloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs = torch.tensor(inputs.permute(0,3,1,2)).float().to(device)
        labels = labels.to(device)
        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = student_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 통계를 출력합니다.
        running_loss += loss.item()
        print('%s=> loss: %.3f' %(i, loss))
        loss_history.append(loss.cpu().detach())

print('Finished Training')

# =============================================================================
# create noisy student data
# =============================================================================
# load student data
pseudo_student_dataset = []
pseudo_student_dataset = create_training_data(1500, 2000, pseudo_student_dataset)
pseudo_student_dataset = random.sample(pseudo_student_dataset, len(pseudo_student_dataset))

# =============================================================================
# Psuedo Labeling
# =============================================================================
new_pseudo_student_dataset = []
for i, data in enumerate(pseudo_student_dataset, 0):
    # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
    inputs, labels = data
    inputs = torch.tensor(inputs).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)

    # create pseudo label
    outputs = teacher(inputs)
    new_label = torch.softmax(outputs, -1)
    y = int(torch.argmax(new_label, dim=1))
    new_pseudo_student_dataset.append([np.array(inputs.squeeze(0).permute(1, 2, 0).cpu().detach().int()), y])

# create psuedo data (append train data with pseudo data)
temp_dataset = student_dataset + new_pseudo_student_dataset
new_psuedo_student_dataloader = data_util.DataLoader(temp_dataset, batch_size=64, shuffle=True)


# =============================================================================
# Train Student Model
# =============================================================================
loss_history = []
## train
for epoch in tqdm(range(epoch)):   # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(new_psuedo_student_dataloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs = torch.tensor(inputs.permute(0,3,1,2)).float().to(device)
        labels = labels.to(device)
        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = student_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 통계를 출력합니다.
        running_loss += loss.item()
        print('%s=> loss: %.3f' %(i, loss))
        loss_history.append(loss.cpu().detach())

print('Finished Training')


# =============================================================================
# Test Self-Training Data
# =============================================================================
y_pred = []
y_true = []
for i, data in enumerate(test_dataset, 0):
    # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
    inputs, labels = data
    inputs = torch.tensor(inputs).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)

    # create pseudo label
    outputs = student_model(inputs)
    new_label = torch.softmax(outputs, -1)
    y = int(torch.argmax(new_label, dim=1))
    y_pred.append(y)
    y_true.append(labels)

print(classification_report(y_true, y_pred, target_names=CATEGORIES))

