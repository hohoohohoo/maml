import torch
import torch.nn as nn
import random
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import time


class MAMLModel_2hidden(nn.Module):
    def __init__(self,in_features,layer_length):
        super(MAMLModel_2hidden, self).__init__()
        self.in_features = in_features
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_features,layer_length)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(layer_length,layer_length)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(layer_length,1))
        ]))
        
    def forward(self, x):
        return self.model(x)
    
    def parameterised(self, x, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        weights = [w.to(self.device) for w in weights]
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        return x

class MAMLModel_3hidden(nn.Module):
    def __init__(self, in_features, layer_length):
        super(MAMLModel_3hidden, self).__init__()
        self.in_features = in_features
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_features, layer_length)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(layer_length, layer_length)),
            ('relu3', nn.ReLU()),
            ('l4', nn.Linear(layer_length, layer_length)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(layer_length, 1))
        ]))
        
    def forward(self, x):
        return self.model(x)
    
    def parameterised(self, x, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = [w.to(self.device) for w in weights]
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[6], weights[7])
        return x

class MAMLModel_4hidden(nn.Module):
    def __init__(self, in_features, layer_length):
        super(MAMLModel_4hidden, self).__init__()
        self.in_features = in_features
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_features, layer_length)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(layer_length, layer_length)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(layer_length, layer_length)),
            ('relu3', nn.ReLU()),
            ('l4', nn.Linear(layer_length, layer_length)),
            ('relu4', nn.ReLU()),
            ('l5', nn.Linear(layer_length, 1))
        ]))
        
    def forward(self, x):
        return self.model(x)
    
    def parameterised(self, x, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = [w.to(self.device) for w in weights]
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[6], weights[7])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[8], weights[9])
        return x

class OptimizedMAML():
    def __init__(self, model, inner_lr, meta_lr, K=5, inner_steps=1,
                 dataset_in=None, dataset_out=None, tasks_per_meta_batch=16,
                 loss_logging_config=None):
        self.train_in = dataset_in
        self.train_out = dataset_out

        # important objects
        self.model = model
        self.weights = list(model.parameters())
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # metrics
        self.plot_every = 10
        self.print_every = 200
        self.meta_losses = []

        # Loss logging configuration
        self.loss_logging_config = loss_logging_config or {}
        self.enable_loss_logging = self.loss_logging_config.get('enabled', False)
        self.loss_log_every = self.loss_logging_config.get('log_every', 1000)
        self.iteration_loss_log = []  # List of (iteration, loss) dicts

        if torch.cuda.is_available():
            self.model.cuda()

    def inner_loop_single_task(self, task_idx):
        """단일 태스크 inner loop - 안정적인 버전"""
        temp_weights = [w.clone() for w in self.weights]
        
        for step in range(self.inner_steps):
            indices = random.sample(range(len(self.train_in[0])), self.K)
            X = self.train_in[task_idx][indices]
            y = self.train_out[task_idx][indices]
            
            # Mixed precision 사용
            with torch.cuda.amp.autocast():
                loss = self.criterion(self.model.parameterised(X, temp_weights), y + 1e-6) / self.K
            
            grad = torch.autograd.grad(loss, temp_weights, create_graph=False)  # FOMAML: 1st order only
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
        
        # Meta-update를 위한 loss 계산
        indices = random.sample(range(len(self.train_in[0])), self.K)
        X = self.train_in[task_idx][indices]
        y = self.train_out[task_idx][indices]
        
        with torch.cuda.amp.autocast():
            loss = self.criterion(self.model.parameterised(X, temp_weights), y + 1e-6) / self.K
        
        return loss

    def main_loop_optimized(self, num_iterations, start_iteration=0):
        """최적화된 메인 루프 - 안정적인 버전

        Args:
            num_iterations: Number of iterations to train
            start_iteration: Starting iteration number (for cumulative tracking)
        """
        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):
            cumulative_iteration = start_iteration + iteration
            meta_losses = []

            # 여러 태스크를 병렬로 처리
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for _ in range(self.tasks_per_meta_batch):
                    task_idx = random.randint(0, len(self.train_in)-1)
                    future = executor.submit(self.inner_loop_single_task, task_idx)
                    futures.append(future)

                for future in futures:
                    meta_losses.append(future.result())

            meta_loss = sum(meta_losses) / len(meta_losses)

            # Meta gradient 계산 및 업데이트
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            self.meta_optimiser.zero_grad()

            # 로그 기록
            current_loss = meta_loss.item()
            epoch_loss += current_loss

            # Loss logging at specified intervals
            if self.enable_loss_logging and cumulative_iteration % self.loss_log_every == 0:
                self.iteration_loss_log.append({
                    'iteration': cumulative_iteration,
                    'loss': current_loss
                })

            if iteration % self.print_every == 0:
                print(f"{iteration}/{num_iterations} | loss: {epoch_loss / self.plot_every:.6f}")
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0

    def main_loop_sequential(self, num_iterations, start_iteration=0):
        """순차적 처리 버전 - 가장 안정적

        Args:
            num_iterations: Number of iterations to train
            start_iteration: Starting iteration number (for cumulative tracking)
        """
        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):
            cumulative_iteration = start_iteration + iteration
            meta_losses = []

            # 순차적으로 태스크 처리
            for _ in range(self.tasks_per_meta_batch):
                task_idx = random.randint(0, len(self.train_in)-1)
                loss = self.inner_loop_single_task(task_idx)
                meta_losses.append(loss)

            meta_loss = sum(meta_losses) / len(meta_losses)

            # Meta gradient 계산 및 업데이트
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            self.meta_optimiser.zero_grad()

            # 로그 기록
            current_loss = meta_loss.item()
            epoch_loss += current_loss

            # Loss logging at specified intervals
            if self.enable_loss_logging and cumulative_iteration % self.loss_log_every == 0:
                self.iteration_loss_log.append({
                    'iteration': cumulative_iteration,
                    'loss': current_loss
                })

            if iteration % self.print_every == 0:
                print(f"{iteration}/{num_iterations} | loss: {epoch_loss / self.plot_every:.6f}")
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0

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
                'inner_lr': self.inner_lr,
                'meta_lr': self.meta_lr,
                'K': self.K,
                'inner_steps': self.inner_steps,
                'tasks_per_meta_batch': self.tasks_per_meta_batch,
                'loss_log_every': self.loss_log_every
            },
            'loss_log': self.iteration_loss_log
        }

        with open(save_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"Loss log saved: {save_path} ({len(self.iteration_loss_log)} entries)") 