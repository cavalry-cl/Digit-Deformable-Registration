import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DigitImageDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = []
        self.labels = []
        for file in os.listdir(path):
            transform = transforms.ToTensor()
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            img = transform(img)
            self.data.append(img)
            self.labels.append(int(file[-5]))
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TrueDigitImages:
    def __init__(self, path):
        self.data = [None for _ in range(10)]
        for file in os.listdir(path):
            self.data[int(file[0])] = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            # transform = transforms.ToTensor()
            transform = transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.1,), (0.3,))
                        ])
            self.data[int(file[0])] = transform(self.data[int(file[0])]).to(device=device)
    def __getitem__(self, idx):
        return self.data[idx]
    
class DigitDataset(Dataset):
    def __init__(self, data, total, labels, size):
        self.target_id = [None for _ in range(10)]
        for i in range(10):
            l = data[i].shape[0]
            self.target_id[i] = torch.randint(0, l, (total.shape[0], size))
        self.l = total.shape[0] * size
        self.size = size
        self.data = data
        self.total = total
        self.labels = labels

    def __len__(self):
        return self.l
    
    def __getitem__(self, idx):
        id = idx % self.total.shape[0]
        s = idx // self.total.shape[0]
        label = self.labels[id]
        target_id = self.target_id[label][id,s]
        return torch.cat((self.total[id], self.data[label][target_id]),dim=0)