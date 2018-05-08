# -*-coding: utf-8 -*-
"""
Created on May 3, 2018
@author: Yue Peng
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data.dataset import Dataset
from sklearn.utils.extmath import softmax
from sklearn.metrics import roc_auc_score
import _data

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

train_data, test_data = _data.data()
cols = train_data.columns.values.tolist()
cols.remove("diabetes")
# Hyper Parameters
num_epochs = 10
batch_size = 16
learning_rate = 1e-3
USE_CUDA = True

# Data Loader (Input Pipeline)
def train_loader(dat, bsz):
    # columns used to train
    cols = dat.columns.values.tolist()
    # remove label
    cols.remove("diabetes")
    X = torch.from_numpy(dat.as_matrix(cols)).double()
    y = torch.LongTensor(dat.as_matrix(["diabetes"]).reshape(-1).tolist())
    train_set = torch.utils.data.TensorDataset(X, y)

    return torch.utils.data.DataLoader(train_set, batch_size=bsz, shuffle=True)

def test_loader(dat, bsz):
    cols = dat.columns.values.tolist()
    for p in ["diabetes"]:
        cols.remove(p)
    X = torch.from_numpy(dat.as_matrix(cols)).double()
    y = torch.LongTensor(dat.as_matrix(["diabetes"]).reshape(-1).tolist())
    test_set = torch.utils.data.TensorDataset(X, y)

    return torch.utils.data.DataLoader(test_set, batch_size=bsz, shuffle=True)


# Neural Network Model (4 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3, hs4, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hs1)
        self.fc2 = nn.Linear(hs1, hs2)
        self.fc3 = nn.Linear(hs2, hs3)
        self.fc4 = nn.Linear(hs3, hs4)
        self.fc5 = nn.Linear(hs4, num_classes)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.95)

    def forward(self, x):
        out = self.fc1(x)
        out = self.prelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.prelu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.prelu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.prelu(out)
        out = self.dropout(out)
        out = self.fc5(out)
        return out


net = Net(input_size=39, hs1=748, hs2=256, hs3=64, hs4=16, num_classes=2)

if USE_CUDA:
    try:
        net = net.cuda()
    except Exception as e:
        print(e)
        USE_CUDA = False
        N_EPOCHS = 2
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    total_loss = 0
    for i, (x, labels) in enumerate(train_loader(train_data, batch_size)):
        # Convert torch tensor to Variable
        x = Variable(x).float()
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(x)
        loss = criterion(outputs, labels)
        total_loss += float(loss.data[0])
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.6f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size,
                     total_loss / (i + 1)))

# Test the Model
correct = 0
total = 0
net.eval()
for x, labels in test_loader(test_data, len(test_data)):
    x = Variable(x).float()
    outputs = net(x)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the %d test images: %d %%'
      % (len(test_data), 100 * correct / total))
probs = softmax(outputs.detach().numpy())[:, 1]
roc_auc_score(test_data.diabetes, probs)

# feature map
fc1 = np.matmul(train_data.as_matrix(cols), net.fc1._parameters["weight"].detach().numpy().T)+\
    net.fc1._parameters["bias"].detach().numpy()
fc2 = np.matmul(fc1, net.fc2._parameters["weight"].detach().numpy().T)+\
    net.fc2._parameters["bias"].detach().numpy()
# 64 new features
fc3 = np.matmul(fc2, net.fc3._parameters["weight"].detach().numpy().T)+\
    net.fc3._parameters["bias"].detach().numpy()
fc4 = np.matmul(fc3, net.fc4._parameters["weight"].detach().numpy().T)+\
    net.fc4._parameters["bias"].detach().numpy()

fc1_test = np.matmul(test_data.as_matrix(cols), net.fc1._parameters["weight"].detach().numpy().T)+\
    net.fc1._parameters["bias"].detach().numpy()
fc2_test = np.matmul(fc1_test, net.fc2._parameters["weight"].detach().numpy().T)+\
    net.fc2._parameters["bias"].detach().numpy()
fc3_test = np.matmul(fc2_test, net.fc3._parameters["weight"].detach().numpy().T)+\
    net.fc3._parameters["bias"].detach().numpy()
fc4_test = np.matmul(fc3_test, net.fc4._parameters["weight"].detach().numpy().T)+\
    net.fc4._parameters["bias"].detach().numpy()
