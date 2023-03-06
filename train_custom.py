import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pickle
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import time
import resnet3d
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


dataset_path = 'data/fighting_dataset.pkl'
label_path = 'data/label.pkl'
with open(dataset_path, 'rb') as f:
    data = pickle.load(f)
with open(label_path, 'rb') as f:
    label_s = pickle.load(f)


# Chia dữ liệu ban đầu thành tập huấn luyện và tập kiểm tra
train_data, test_data, train_labels, test_labels = train_test_split(data, label_s, test_size=0.2)

# Chia tập huấn luyện thành tập huấn luyện và tập xác nhận
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2)

# Chuyển dữ liệu sang kiểu Tensor
train_data = torch.stack(train_data)
train_labels = torch.stack(train_labels)
val_data = torch.stack(val_data)
val_labels = torch.stack(val_labels)
test_data = torch.stack(test_data)
test_labels = torch.stack(test_labels)

# Tạo DataLoader cho tập huấn luyện
train_dataset = TensorDataset(train_data, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Tạo DataLoader cho tập xác nhận
val_dataset = TensorDataset(val_data, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Tạo DataLoader cho tập kiểm tra
test_dataset = TensorDataset(test_data, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)


#####################################################

model = resnet3d.ResNet(resnet3d.BasicBlock, [1, 1, 1, 1], [32, 128, 256, 512])
model.to(device)
checkpoint = 'model.pth' 

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

batch_size = 4
display_step = 20

loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []
best_val_loss = 999

print('start training')
epochs = 10

for epoch in range(epochs):

    # Quá trình training 
    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)

        # Clear gradients for this training step 
        optimizer.zero_grad()
        output = model(data)

        # Backpropagation, compute gradients
        loss = criterion(output, target.float())
        loss.backward()

        # Apply gradients
        optimizer.step()
        if batch_idx % display_step == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader), loss.item()))
            
    # Quá trình testing 
    model.eval()
    test_loss = 0
    correct = 0
    # Set no grad cho quá trình testing
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += criterion(output, target.float()) 
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.argmax(dim = 1, keepdim = True).view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset) 
    if test_loss < best_val_loss:
      best_val_loss = test_loss
      torch.save(model.state_dict(), checkpoint)  # Lưu model path
    print("***********    TEST_ACC = {:.2f}%    ***********".format(correct/100))