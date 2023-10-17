import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import math
import time



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parameters 
batch_size = 100
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


train_dataset = torchvision.datasets.MNIST(root='../../data', train=True,download=True, transform=transforms.ToTensor())
                              

test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
x,y = train_dataset[0]



#Neural Network 
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.leakyrelu(out)
        out = self.fc2(out)
        return out

model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

class NeuralNetwork2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork2, self).__init__()
        self.fC1 = nn.Linear(input_size, 600) 
        self.relu1 = nn.ReLU()
        self.fC2 = nn.Linear(600, 300)
        self.relu2 = nn.ReLU()
        self.fC3 = nn.Linear(300, num_classes)
    
    def forward(self, x):
        out = self.fC1(x)
        out = self.relu1(out)
        out = self.fC2(out)
        out = self.relu2(out)
        out = self.fC3(out)
        return out



model2 = NeuralNetwork2(input_size, num_classes).to(device)


class NeuralNetwork3(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork3, self).__init__()
        self.fC1 = nn.Linear(input_size, 4096) 
        self.relu1 = nn.ReLU()
        self.fC2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fC3 = nn.Linear(4096, 512)
        self.relu3 = nn.ReLU()
        self.fC4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.fC5 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        out = self.fC1(x)
        out = self.relu1(out)
        out = self.fC2(out)
        out = self.relu2(out)
        out = self.fC3(out)
        out = self.relu3(out)
        out = self.fC4(out)
        out = self.relu4(out)
        out = self.fC5(out)
        return out

model3 = NeuralNetwork3(input_size, num_classes).to(device)

#Set Loss
criterion=nn.CrossEntropyLoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

time1=time.time()


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

       
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
time2=time.time()
print('Time Passed: {:.3f}sec'.format(time2-time1))
# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))






