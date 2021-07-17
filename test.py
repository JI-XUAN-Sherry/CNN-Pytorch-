import torch
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms    # torchvision contains common utilities for computer vision
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import cv2

from model import *
from data import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

def load_data(test_batch_size, test_number_work):

    PATH = 'save2.pt'
    checkpoint = torch.load(PATH)

    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Fetch test data
    path_val_test = "/home/user1/project/project/Dataset/Frame31/"

    test_trainsform = transforms.Compose([transforms.Resize((80, 80)),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    test_loader = DataLoader(MyDataset(path_val_test,test_trainsform),
                             batch_size = test_batch_size,
                             #shuffle=True,
                             num_workers = test_number_work)

    return (test_loader)

def find_chips_rejion(chips_list, img_root):
    length = len(chips_list)
    img = cv2.imread(img_root)
    w = 80
    h = 80
    color = (0, 255, 255)
    for i in range(length):
        x = 80*(chips_list[i]%16)
        y = 80*int(chips_list[i]/16)
        cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=3)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test(model, test_loader):
    
    model.eval()

    # test_loss = 0
    # correct = 0

    with torch.no_grad():
        testnum=0
        chips_list = []
        for image in test_loader: 

            image = image.to(device)
            # target = target.to(device)
            # image = image.view(1, -1)

            # Retrieve output
            output = model(image)

            if output[0, 0]<output[0, 1]:
                chips_list.append(testnum)
                print(output)
            testnum+=1

            # # Calculate & accumulate loss
            # test_loss += F.nll_loss(output, target, reduction='sum').data.item()

            # pred = output.image.argmax(1)

            # correct += pred.eq(target.data).sum()
        print("chips list", chips_list)
        return chips_list
    # # Print out average test loss
    # # test_loss /= len(test_loader.dataset)
    # print('\nhave chips: {:.4f}\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset))) 

if __name__ == '__main__':

    model = Net().to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.0001)

    # Load data
    test_loader = load_data(1, 4)


    # Train & test the model
    chips_list = test(model, test_loader)


    root = "/home/user1/project/project/Dataset/Frame31.jpg"
    find_chips_rejion(chips_list, root)
