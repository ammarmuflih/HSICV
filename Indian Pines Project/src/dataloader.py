import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from scipy.io import loadmat

class CustomDataset(Dataset):
    def __init__(self, datapath, gtpath):
        self.datapath = datapath
        self.gtpath = gtpath
        self.data = []
        self.gt = []

        with rasterio.open(self.datapath) as src:
            data = src.read()
        
        mat = loadmat(self.gtpath)
        gt = mat['indian_pines_gt']

        data = data.transpose(1, 2, 0)

        mask = (gt != 0)
        self.data = data[mask]
        self.gt =  gt[mask]
    
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.gt[index]

        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return sample_tensor, label_tensor
