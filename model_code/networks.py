from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
from torch.autograd import Variable

NetIO = Union[FloatTensor, Variable]


class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: NetIO) -> NetIO:
        # compute the representation for each data point
        x = self.phi.forward(x)

        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
       #x = torch.sum(x, dim=0, keepdim=True)
        x = torch.mean(x, dim=0, keepdim=True)
        # compute the output
        out = self.rho.forward(x)

        return out

class changed_MLP(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module, in_1_size:int = 1):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.in_1_size=in_1_size
    def forward(self,x):
        y=self.phi.forward(x[:,0:self.in_1_size])
        x2=x[:,self.in_1_size:].clone().detach() 
        x2=torch.concat((y,x2),dim=1)
        y=self.rho.forward(x2)
        return y

class FirstMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 40)
        #self.fc1_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(40, self.output_size)
        #self.bn = nn.BatchNorm1d(30)
        self.ln = nn.LayerNorm(40)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight.data)
                 
    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(self.ln(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SecondMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ln = nn.LayerNorm(40)
        self.fc1 = nn.Linear(self.input_size, 40)
        self.fc2 = nn.Linear(40, self.output_size)

    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(self.ln(self.fc1(x)))
        #x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 40)
        self.fc2 = nn.Linear(40,40)
        self.fc4 = nn.Linear(40,40)
        self.fc3 = nn.Linear(40, self.output_size)
        self.dropout = nn.Dropout(0.3)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight.data)
    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.relu(self.fc4(x))
        #x = F.relu(self.ln(self.fc5(x)))
        x = self.fc3(x)
        #x = self.fc4(x)
        return x


class MLP_Aadam(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, self.output_size)
        self.dropout = nn.Dropout(0.3)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight.data)
    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.relu(self.fc4(x))
        #x = F.relu(self.ln(self.fc5(x)))
        x = self.fc3(x)
        #x = self.fc4(x)
        return x