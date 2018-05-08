import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
from sklearn.utils.extmath import softmax
from sklearn.metrics import roc_auc_score
import _data
import warnings
warnings.filterwarnings("ignore")

train_data, test_data = _data.data()
new_cols = feature_score.iloc[0:18,:]["Feature"]
trainD = pd.concat([train_data[new_cols], train_data[new_cols]], axis=1).as_matrix()
trainD = np.concatenate((trainD, fc1), axis=1)
trainD = trainD.reshape((-1, 1, 28, 28))

testD = pd.concat([test_data[new_cols], test_data[new_cols]], axis=1).as_matrix()
testD = np.concatenate((testD, fc1_test), axis=1)
testD = testD.reshape((-1, 1, 28, 28))
# ====================== CNN ======================== #
# Hyper Parameters
num_epochs = 3
batch_size = 16
learning_rate = 0.0001



def data_loader(dat, bsz):
    # columns used to train

    X = torch.from_numpy(dat).double()
    if len(dat) > 10000:
        y = torch.LongTensor(train_data.as_matrix(["diabetes"]).reshape(-1).tolist())
    else:
        y = torch.LongTensor(test_data.as_matrix(["diabetes"]).reshape(-1).tolist())
    data_set = torch.utils.data.TensorDataset(X, y)

    return torch.utils.data.DataLoader(data_set, batch_size=bsz, shuffle=True)



class CNN(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x



cnn = CNN()
cnn.double()
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)
# Train the Model
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(data_loader(trainD, batch_size)):
        images = images.double()
        images = Variable(images)
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.6f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, total_loss / (i + 1)))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0

for images, labels in data_loader(testD, len(testD)):
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()


print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

roc_auc_score(test_data["diabetes"], softmax(outputs.detach().numpy())[:, 1])


# conv2 feature map
conv1 = cnn.conv1.forward(Variable(next(iter(data_loader(
        trainD, len(trainD))))[0])).detach().numpy()
conv1 = conv1.reshape((-1, 10, 24*24))
conv1 = np.mean(conv1, axis=1)

conv1_test = cnn.conv1.forward(Variable(next(iter(data_loader(
        testD, len(testD))))[0])).detach().numpy()
conv1_test = conv1_test.reshape((-1, 10, 24*24))
conv1_test = np.mean(conv1_test, axis=1)


conv2 = cnn.conv2.forward(
    cnn.conv1.forward(Variable(next(iter(data_loader(
        trainD, len(trainD))))[0]))).detach().numpy()

conv2_test = cnn.conv2.forward(
    cnn.conv1.forward(Variable(next(iter(data_loader(
        testD, len(testD))))[0]))).detach().numpy()

# calculate mean of all maps
conv2 = np.mean(conv2, axis=2).reshape(len(trainD), -1)
conv2_test = np.mean(conv2_test, axis=2).reshape(len(testD), -1)

