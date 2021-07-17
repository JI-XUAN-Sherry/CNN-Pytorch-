from torch.utils.data import Dataset
from PIL import Image
  
import glob
import os

class MyDataset(Dataset):
    def __init__(self,root,transform):
        self.root = root
        self.transform = transform
        self.file = sorted(glob.glob(os.path.join(root,'*.jpg')))

    def __getitem__(self,index):
        img = Image.open(self.file[index])
        item = self.transform(img)
        return item

    def __len__(self):
        return len(self.file)
