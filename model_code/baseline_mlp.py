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
        self.fc3 = nn.Linear(40,40)
        self.fc4 = nn.Linear(40, self.output_size)
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
        x = F.relu(self.fc3(x))
        #x = F.relu(self.ln(self.fc5(x)))
        x = self.fc4(x)
        #x = self.fc4(x)
        return x


class MLP_Aadam(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, self.output_size)
        self.dropout = nn.Dropout(0.3)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x: NetIO) -> NetIO:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLP_pretraining:
    """
    MLP Pretraining wrapper class that handles training loop and checkpoint saving.

    Args:
        lr (float): Learning rate
        wd (float): Weight decay
        dataset_in (Tensor): Input dataset [N, seq_len, features]
        dataset_out (Tensor): Output dataset [N, seq_len, 1]
        iteration (int): Number of training iterations
        hidden_size (int): Hidden layer size (40 for MLP, 256 for MLP_Aadam)
        input_size (int): Input feature size
    """
    def __init__(self, lr, wd, dataset_in, dataset_out, iteration, hidden_size=40, input_size=9,
                 loss_logging_config=None):
        self.lr = lr
        self.wd = wd
        self.dataset_in = dataset_in
        self.dataset_out = dataset_out
        self.iteration = iteration
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Create model based on hidden size
        self.model = self._create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.criterion = nn.MSELoss()

        # Loss logging configuration
        self.loss_logging_config = loss_logging_config or {}
        self.enable_loss_logging = self.loss_logging_config.get('enabled', False)
        self.loss_log_every = self.loss_logging_config.get('log_every', 1000)
        self.iteration_loss_log = []  # List of (iteration, loss) dicts

    def _create_model(self):
        """Create MLP model with specified hidden size"""
        class FlexibleMLP(nn.Module):
            def __init__(inner_self, input_size, hidden_size, output_size=1):
                super().__init__()
                inner_self.fc1 = nn.Linear(input_size, hidden_size)
                inner_self.fc2 = nn.Linear(hidden_size, hidden_size)
                inner_self.fc3 = nn.Linear(hidden_size, hidden_size)
                inner_self.fc4 = nn.Linear(hidden_size, output_size)
                # FAIRINIT: rely on nn.Linear default (kaiming_uniform_(a=sqrt(5))), same as MAMLModel_3hidden

            def forward(inner_self, x):
                x = F.relu(inner_self.fc1(x))
                x = F.relu(inner_self.fc2(x))
                x = F.relu(inner_self.fc3(x))
                x = inner_self.fc4(x)
                return x

        device = self.dataset_in.device
        return FlexibleMLP(self.input_size, self.hidden_size).to(device)

    def loop(self, checkpoint_dir=None, checkpoint_interval=10000, start_iteration=0):
        """
        Training loop

        Args:
            checkpoint_dir (str): Directory to save checkpoints
            checkpoint_interval (int): Save checkpoint every N iterations
            start_iteration (int): Starting iteration number (for cumulative tracking)

        Returns:
            float: Final training loss
        """
        import os
        import random

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        num_tasks = len(self.dataset_in)
        self.model.train()

        running_loss = 0.0
        final_loss = 0.0

        for i in range(self.iteration):
            cumulative_iteration = start_iteration + i + 1

            # Random task sampling
            task_idx = random.randint(0, num_tasks - 1)
            X = self.dataset_in[task_idx]  # [seq_len, features]
            y = self.dataset_out[task_idx]  # [seq_len] or [seq_len, 1]

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X)  # [seq_len, 1]

            # Ensure shapes match to avoid broadcasting issues
            if predictions.dim() == 2 and y.dim() == 1:
                predictions = predictions.squeeze(-1)  # [seq_len, 1] -> [seq_len]
            elif predictions.dim() == 1 and y.dim() == 2:
                y = y.squeeze(-1)  # [seq_len, 1] -> [seq_len]

            loss = self.criterion(predictions, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            running_loss += current_loss

            # Loss logging at specified intervals
            if self.enable_loss_logging and cumulative_iteration % self.loss_log_every == 0:
                self.iteration_loss_log.append({
                    'iteration': cumulative_iteration,
                    'loss': current_loss
                })

            # Logging
            if (i + 1) % 1000 == 0:
                avg_loss = running_loss / 1000
                print(f"Iteration {i+1}/{self.iteration}, Loss: {avg_loss:.6f}")
                final_loss = avg_loss
                running_loss = 0.0

            # Checkpoint saving
            if checkpoint_dir and (i + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i+1}.pth")
                torch.save({
                    'iteration': i + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': current_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        return final_loss

    def save_loss_log(self, save_path):
        """Save iteration loss log to JSON file

        Args:
            save_path: Path to save the loss log JSON file
        """
        import json
        import os

        if not self.iteration_loss_log:
            print("No loss log entries to save")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        log_data = {
            'config': {
                'lr': self.lr,
                'wd': self.wd,
                'hidden_size': self.hidden_size,
                'loss_log_every': self.loss_log_every
            },
            'loss_log': self.iteration_loss_log
        }

        with open(save_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"Loss log saved: {save_path} ({len(self.iteration_loss_log)} entries)")