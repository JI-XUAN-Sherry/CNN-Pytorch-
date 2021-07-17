import torch
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms    # torchvision contains common utilities for computer vision
import torch.utils.data as data

from model import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

def load_data(train_batch_size, train_number_work):
    
    # Fetch training data
    path_val_train = '/home/user1/project/project/Dataset/trainData_Binary/'
    # path_val_train = '/home/user1/project/project/Dataset/trainData/'

    train_trainsform = transforms.Compose([transforms.Resize((80, 80)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5], [0.5])])

    train = datasets.ImageFolder(path_val_train, transform = train_trainsform)
    train_loader = data.DataLoader(train, batch_size = train_batch_size, shuffle = True, num_workers = train_number_work)

    return (train_loader)
    
def train(model, optimizer, epoch, train_loader, log_interval):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = data.to(device)
        target = target.to(device)
        
        # Clear the gradients, since PyTorch accumulates them
        optimizer.zero_grad()

        # Forward propagation
        output = model(data)

        # Calculate negative log likelihood loss
        loss = F.nll_loss(output, target)

        # Backward propagation
        loss.backward()

        # Update the gradients
        optimizer.step()

        # Output debug message
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

    # Save the model for future use
    torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'loss':loss},
                "save2.pt")

if __name__ == '__main__':

    model = Net().to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.0001)

    # Load data
    train_loader = load_data(200, 4)

    # Train & test the model
    for epoch in range(0, 50):
        train(model, optimizer, epoch, train_loader, log_interval=1)
