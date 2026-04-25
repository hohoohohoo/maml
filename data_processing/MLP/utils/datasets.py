from typing import Tuple

import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

class libdata():
    def __init__(self,input):
        self.input = input
        X = [torch.tensor(d["input"], dtype=torch.float32) for d in input]
        y = [torch.tensor(d["output"], dtype=torch.float32) for d in input]
        # Step 2: stack into single tensor
        X_tensor = torch.stack(X)  # shape: [N, input_dim]
        y_tensor = torch.stack(y)  # shape: [N, output_dim]
        self.X = X_tensor
        self.Y = y_tensor
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
