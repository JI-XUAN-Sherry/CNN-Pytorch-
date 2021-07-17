import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Create CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution 1 , input_shape=(3,80,80)
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0) #output_shape=(16,78,78) #(80-3+1)/1 #(weigh-kernel+1)/stride 無條件進位
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,39,39) #(78/2)
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0) #output_shape=(32,37,37)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,18,18) #(37/2)
        
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0) #output_shape=(16,16,16)
        self.relu3 = nn.ReLU() # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,8,8) #(16/2)

        # Fully connected 1 ,#input_shape=(8*12*12)
        self.fc1 = nn.Linear(16 * 8 * 8, 10) 
        self.relu5 = nn.ReLU() # activation
        self.fc2 = nn.Linear(10, 2) 
        self.output = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        x = self.cnn1(x) # Convolution 1
        x = self.relu1(x)
        x = self.maxpool1(x)# Max pool 1
        x = self.cnn2(x) # Convolution 2
        x = self.relu2(x) 
        x = self.maxpool2(x) # Max pool 2
        x = self.cnn3(x) # Convolution 3
        x = self.relu3(x)
        x = self.maxpool3(x) # Max pool 3
        x = x.view(x.size(0), -1) # last CNN faltten con. Linear NN
        x = self.fc1(x) # Linear function (readout)
        x = self.fc2(x)
        x = self.output(x)

        return x
