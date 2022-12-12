#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import OrderedDict
from pathlib import Path
from time import time
from typing import Tuple

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.models import resnet18
import pandas as pd
import numpy as np


# In[ ]:


DATA_ROOT = Path("./data")


# In[ ]:


"""Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        x = torch.flatten(x, 1)
#         self.conv1 = nn.Conv2d(1, 32, (25,1))
#         self.pool = nn.MaxPool2d(1, 20)
#         self.conv2 = nn.Conv2d(32, 32, (25,1))
        self.fc1 = nn.Linear(520, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


# In[ ]:


"""complex CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

class CNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, (25,1))
        self.pool = nn.MaxPool2d(1, 20)
        self.conv2 = nn.Conv2d(64, 128, (25,1))
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


# In[ ]:


def load_model(model_name: str) -> nn.Module:
    if model_name == "Net":## put in Simple ConvNet
        return Net()
    elif model_name == "CNet":## put in Complex ConvNet over here
        return CNet()
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")


# In[ ]:


def load_cifar(download=False) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=True, download=download, transform=transform
    )
    testset = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=False, download=download, transform=transform
    )
    return trainset, testset


# In[ ]:


def load_data() -> Tuple[torch.utils.data.DataLoader,torch.utils.data.DataLoader]:
    ## Load UJI train dataset
    path = "C:/Users/vishalsh/Desktop/ESML/Flower/uji/UJIndoorLoc/trainingData.csv"
    df_train = pd.read_csv(path)
    trainset_to_dataloader, testset_to_dataloader = preprocess_data(df_train)
#     trainsetloader = DataLoader(dataset=train_set_to_dataloader, batch_size=16, shuffle=True)
#     testsetloader = DataLoader(dataset=test_set_to_dataloader, batch_size=16, shuffle=False)
    return trainset_to_dataloader, trainset_to_dataloader


# In[ ]:


def preprocess_data(df):
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.9
    df_train = df[msk]
    df_test = df[~msk]
    df_train,df_test = makelabels(df_train,df_test)
    X_train, y_train, X_test, y_test = slice_dataframe(df_train,df_test) 
    trainset = processRSS(X_train)
    testset  = processRSS(X_test)
    train_set_to_dataloader = createDataset(dataframe = trainset)
    test_set_to_dataloader = createDataset(dataframe = testset)
    return train_set_to_dataloader,test_set_to_dataloader


# In[ ]:


def makelabels(df_train,df_test):
    blds = np.asarray(pd.get_dummies(df_train['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(df_train['FLOOR']))
#     rpos = np.asarray(pd.get_dummies(df_train['RELATIVEPOSITION']))
# spid = np.asarray(pd.get_dummies(df_train['SPACEID']))
    label_train = np.concatenate((blds, flrs), axis=1).tolist()
    blds = np.asarray(pd.get_dummies(df_test['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(df_test['FLOOR']))
#     rpos = np.asarray(pd.get_dummies(df_test['RELATIVEPOSITION']))
# spid = np.asarray(pd.get_dummies(df_test['SPACEID']))
    label_test = np.concatenate((blds, flrs), axis=1).tolist()
    df_train['label'] = label_train
    df_test['label'] = label_test
    return df_train,df_test

def slice_dataframe():
    X_train = df_train.iloc[:,:520]
    X_test = df_test.iloc[:,:520]
    y_train = df_train.iloc[:,520:531]
    y_test = df_test.iloc[:,520:531]
    return X_train, y_train, X_test, y_test

def processRSS(data):
    # Input:
    # - data: a df of input RSS features in UJIndoorLoc format (-98:min, 0:max, 100:null)
    # Output:
    # - a df of RSS values (0:mull, 1:min, 98:max)
    outOfRange = 100 #null values to be replaced
    weakestSignal = -98 #replaces the null values
    # Change null value to new value and set all lower values to it
    data.replace(outOfRange, weakestSignal, inplace=True)
    data[data < weakestSignal] = weakestSignal
    data = data + 98
    # Input:
    # - data: a df of input RSS features in format (0:mull, 1:min, 98:max)
    # Output:
    # - a df of scaled RSS values [0,1]
    # Normalize data between 0 and 1 where 1 is strong signal and 0 is null
    return data/data.values.max()

def make_dataframe():
    WAP_train = trainset.loc[0:,:'WAP520'].values.tolist()
    df_train['WAP'] = WAP_train
    train_set = df_train.loc[:,["WAP","label"]]
    WAP_test = testset.loc[0:,:'WAP520'].values.tolist()
    df_test['WAP'] = WAP_test
    test_set = df_test.loc[:,["WAP","label"]]
    return train_set, test_set


# In[ ]:


class createDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataframe.shape[0]
        
    def __getitem__(self, index):
        image = self.dataframe.iloc[index]["WAP"]
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transform(image)
        label = self.dataframe.iloc[index]["label"]
        return {"WAP": image , "label": torch.tensor(label, dtype=torch.float)}


# In[ ]:


def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data['WAP'].to(device), data['label'].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.5f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.BCELoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data['WAP'].to(device), data['label'].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            outputs = np.round(outputs)
            total += labels.size(0)
            for i in range(len(np.array((outputs)))):
                if torch.equal(outputs[i],labels[i]):
                    correct = correct + 1
#             print(correct,total)
    accuracy = correct / total
    return loss, accuracy

