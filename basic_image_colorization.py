# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 18:59:45 2022

@author: WooYoungHwang
"""

import os
import cv2
import torch
import random
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data_util
import torchvision.transforms as T

from torch import nn, optim
from torchvision import transforms, datasets
from sklearn.metrics import classification_report

'''
source: https://github.com/keon/3-min-pytorch/blob/master/06-%EC%82%AC%EB%9E%8C%EC%9D%98_%EC%A7%80%EB%8F%84_%EC%97%86%EC%9D%B4_%ED%95%99%EC%8A%B5%ED%95%98%EB%8A%94_%EC%98%A4%ED%86%A0%EC%9D%B8%EC%BD%94%EB%8D%94/denoising_autoencoder.py
'''
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
train_dataloader = data_util.DataLoader(training_dataset, batch_size=32, shuffle=True)

# =============================================================================
# Autoencoder
# =============================================================================
EPOCH = 100

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # self.encoder_fc = nn.Sequential(
        #     nn.Linear(96, 10),
        #     nn.ReLU()
        #     )
        
        # self.decoder_fc = nn.Sequential(
        #     nn.Linear(10, 96),
        #     nn.ReLU()
        #     )
        
        # self.unflatten = nn.Unflatten(1, (6, 4, 4))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded

# Hyperparameter
IC_autoencoder = Autoencoder().to(device)
optimizer = torch.optim.Adam(IC_autoencoder.parameters(), lr = 0.005)
criterion = nn.MSELoss()

def gray(img):
    transform = T.Grayscale(num_output_channels=1)
    noisy_img = transform(img)
    
    return noisy_img


# =============================================================================
# Pretrain Context autoencoder 
# =============================================================================
def train(autoencoder, train_loader):
    autoencoder.train()
    avg_loss = 0
    for step, (x, label) in enumerate(train_loader, 0):

        noisy_x = gray(x.permute(0, 3, 1, 2))  # 입력에 노이즈 더하기

        noisy_x = noisy_x.type(torch.FloatTensor)/255

        y = x.type(torch.FloatTensor)/255
        y = y.to(device)
        
        label = label.to(device)

        encoded, decoded = autoencoder(noisy_x.to(device))

        loss = criterion(decoded, y.permute(0, 3, 1, 2))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        
    return avg_loss / len(train_loader)


for epoch in range(1, EPOCH+1):
    loss = train(IC_autoencoder, train_dataloader)
    print("[Epoch {}] loss:{}".format(epoch, loss))


# =============================================================================
# Visualize Context Autoencoder result
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
test_dataset = create_test_data(0, 1000, test_dataset)
test_dataset = random.sample(test_dataset, len(test_dataset))

test_dataloader = data_util.DataLoader(test_dataset, batch_size=64, shuffle=True)
data, label = next(iter(test_dataloader))
sample_data = data[0]
sample_data = sample_data.type(torch.FloatTensor)/255

# 이미지를 mask로 오염시킨 후, 모델에 통과시킵니다.
original_x = sample_data
noisy_x = gray(original_x.permute(2, 0, 1)).to(device)
_, recovered_x = IC_autoencoder(noisy_x.view(1, 1, 32, 32))
recovered_x = recovered_x.squeeze(0)

# Visualize
f, a = plt.subplots(1, 3, figsize=(15, 15))

# 시각화를 위해 넘파이 행렬로 바꿔줍니다.
original_img = np.reshape(original_x.to("cpu").data.numpy(), (32, 32, 3))
noisy_img = np.reshape(noisy_x.to("cpu").data.numpy(), (32, 32, 1))
recovered_img = recovered_x.to("cpu").permute(1, 2, 0).data.numpy()

# 원본 사진
a[0].set_title('Original')
a[0].imshow((original_img*255).astype(np.uint8))

# 오염된 원본 사진
a[1].set_title('Noisy')
a[1].imshow((noisy_img*255).astype(np.uint8))

# # 복원된 사진
a[2].set_title('Recovered')
a[2].imshow((recovered_img*255).astype(np.uint8))


# # Save Model
save_module_PATH = r'C:\Users\WooYoungHwang\Desktop\SPS\2022 spring semester\다변량통계분석및데이터마이닝\과제\과제7\코드'           #모듈 저장 위치 지정
torch.save(IC_autoencoder.state_dict(), save_module_PATH + 'cifar10_colorization_ae.pt')

# # Load Model
load_module_PATH = r'C:\Users\WooYoungHwang\Desktop\SPS\2022 spring semester\다변량통계분석및데이터마이닝\과제\과제7\코드'  
IC_autoencoder = Autoencoder()
IC_autoencoder.load_state_dict(torch.load(load_module_PATH + 'cifar10_colorization_ae.pt'))
IC_autoencoder = IC_autoencoder.to(device)


# =============================================================================
# Downstream Task
# =============================================================================
class DownstreamTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

downstream = DownstreamTask().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(downstream.parameters(), lr=0.0001)
epoch = 100

for epoch in range(1, epoch+1):

    downstream.train()
    avg_loss = 0
    for step, (x, label) in enumerate(train_dataloader, 0):

        noisy_x = gray(x.permute(0, 3, 1, 2))  # 입력에 노이즈 더하기
        noisy_x = noisy_x.type(torch.FloatTensor)/255
        y = x
        y = y.type(torch.FloatTensor)/255
        y = y.to(device)
        label = label.to(device)
        encoded, _ = IC_autoencoder(noisy_x.to(device))
        output = downstream(encoded)
        
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        avg_loss = avg_loss / len(train_dataloader)

    print("[Epoch {}] loss:{}".format(epoch, avg_loss))

# # Save Model
save_module_PATH = r'C:\Users\WooYoungHwang\Desktop\SPS\2022 spring semester\다변량통계분석및데이터마이닝\과제\과제7\코드'           #모듈 저장 위치 지정
torch.save(downstream.state_dict(), save_module_PATH + 'cifar10_colorization_ae_downstream.pt')

# # Load Model
load_module_PATH = r'C:\Users\WooYoungHwang\Desktop\SPS\2022 spring semester\다변량통계분석및데이터마이닝\과제\과제7\코드'  
downstream = DownstreamTask()
downstream.load_state_dict(torch.load(load_module_PATH + 'cifar10_colorization_ae_downstream.pt'))
downstream = downstream.to(device)


# =============================================================================
# Test Downstream Task
# =============================================================================
y_pred = []
y_true = []
for step, (x, label) in enumerate(test_dataset, 0):
    x = torch.tensor(x).float()/255
    noisy_x = gray(x.permute(2, 0, 1))  # 입력에 노이즈 더하기

    encoded, decoded = IC_autoencoder(noisy_x.unsqueeze(0).to(device))
    output = downstream(encoded)
    new_label = torch.softmax(output, -1)
    y = int(torch.argmax(new_label, dim=1))
    y_pred.append(y)
    y_true.append(label)

print(classification_report(y_true, y_pred, target_names=CATEGORIES))
