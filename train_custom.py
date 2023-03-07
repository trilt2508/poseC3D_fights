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

batch_size = 16
display_step = 20
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


def track_to_tensor(track_path):
    with open(track_path, 'rb') as f:
        track = pickle.load(f)

    return track


def batch_to_tensor(batch):
    tensor_list = []
    for track_path in batch:
        track_tensor = track_to_tensor(track_path)
        tensor_list.append(track_tensor)
    batch_tensor = torch.stack(tensor_list)
    return batch_tensor

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


with open('path_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
    f.close()




# Chia dữ liệu ban đầu thành tập huấn luyện và tập kiểm tra
train_data, test_data, train_labels, test_labels = train_test_split(data[0], data[1], test_size=0.2)

# Chia tập huấn luyện thành tập huấn luyện và tập xác nhận
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2)


# Tạo DataLoader cho tập huấn luyện
train_dataset = MyDataset(train_data, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Tạo DataLoader cho tập xác nhận
val_dataset = MyDataset(val_data, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Tạo DataLoader cho tập kiểm tra
test_dataset = MyDataset(test_data, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

size_train = len(train_dataset)
size_val = len(val_dataset)
size_test = len(test_dataset)
#####################################################

model = resnet3d.ResNet(resnet3d.BasicBlock, [1, 1, 1, 1], [32, 128, 256, 512])
model.to(device)
checkpoint = 'model2.pth' 

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))



loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []
best_val_loss = 999

print('start training')
epochs = 100





for epoch in range(epochs):

    model.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = batch_to_tensor(data)
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
            
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            data = batch_to_tensor(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target.float()) 
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.argmax(dim = 1, keepdim = True).view_as(pred)).sum().item()
    val_loss /= len(val_dataloader.dataset) 
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), checkpoint)  # Lưu model path
    print("***********    VAL_ACC = {:.2f}%    ***********".format(correct/size_val))



#test
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_dataloader:
        data = batch_to_tensor(data)
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target.float()) 
        pred = output.argmax(dim = 1, keepdim = True)
        correct += pred.eq(target.argmax(dim = 1, keepdim = True).view_as(pred)).sum().item()
test_loss /= len(test_dataloader.dataset)
print("***********    VAL_ACC = {:.2f}%    ***********".format(correct/size_test))