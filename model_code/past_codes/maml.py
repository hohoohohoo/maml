import torch
import torch.nn as nn
import random
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_input = torch.tensor(0)
data_output = torch.tensor(0)
testdata_input = torch.tensor(0)
testdata_output = torch.tensor(0)

class MAMLModel_3hidden(nn.Module):
    def __init__(self,in_features,layer_length):
        super(MAMLModel_3hidden, self).__init__()
        self.in_features = in_features
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_features,layer_length)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(layer_length,layer_length)),
            ('relu3', nn.ReLU()),
            ('l4', nn.Linear(layer_length,layer_length)),
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
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[6], weights[7])
        return x
    
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
    

class  MAML():
    def __init__(self, model, inner_lr, meta_lr, K=5, inner_steps=5, dataset_in=data_input, dataset_out=data_output,train_in_test =testdata_input,train_out_test=testdata_output,  tasks_per_meta_batch=30):
        self.train_in = dataset_in
        self.train_out = dataset_out
        self.train_in_test = train_in_test
        self.train_out_test = train_out_test
        
        # important objects
        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
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

        if torch.cuda.is_available():
            self.model.cuda()

    def inner_loop(self):
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]

        # perform training on data sampled from task
        random1 = random.randint(0,len(self.train_in)-1)
        indices = random.sample(range(len(self.train_in[0])), self.K)
        X = self.train_in[random1][indices]
        y = self.train_out[random1][indices]


        #X, y = X.cuda(), y.cuda()
        # Ensure both data and model are on the same device
        #if torch.cuda.is_available():
        #    X, y = X.cuda(), y.cuda()
        for step in range(self.inner_steps):  # inner_steps 만큼 gradient descent 진행
            loss = self.criterion(self.model.parameterised(X, temp_weights), y + 1e-6) / self.K
            # compute grad and update inner loop weights
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
    
        # sample new data for meta-update and compute loss
        indices = random.sample(range(len(self.train_in[0])), self.K)
        X = self.train_in[random1][indices]
        y = self.train_out[random1][indices]
        #X, y = X.cuda(), y.cuda()
        # Ensure both data and model are on the same device
    #    if torch.cuda.is_available():
    #        X, y = X.cuda(), y.cuda()
        loss = self.criterion(self.model.parameterised(X, temp_weights), y+ 1e-6) / self.K

        return loss
    
    def main_loop(self, num_iterations):
        epoch_loss = 0
    # streams: 각 GPU에서 병렬 실행될 CUDA streams
        streams = [torch.cuda.Stream() for _ in range(8)]
        for iteration in range(1, num_iterations + 1):
            meta_losses = []

            def process_one_task( stream):
                with torch.cuda.stream(stream):  # CUDA stream을 사용해 병렬 실행
                    return self.inner_loop()

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(process_one_task,streams[i%8]) 
                    for i in range(self.tasks_per_meta_batch)  # 각 GPU에 할당된 작업
                ]
                for future in futures:
                    meta_losses.append(future.result())
            torch.cuda.synchronize()
            meta_loss = sum(meta_losses) / len(meta_losses)

            # meta gradient 계산
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # 로그 기록
            epoch_loss += meta_loss.item()
            if iteration % self.print_every == 0:
                print(f"{iteration}/{num_iterations} | loss: {epoch_loss / self.plot_every:.6f}")
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0

    def eval(self):
        running_loss = 0.0
        with torch.no_grad():
            self.model.eval()
            for j in range(len(self.train_out_test)):
                x_sampled = self.train_in_test[j]
                x_sampled = x_sampled.unsqueeze(0)
                y_sampled = self.train_out_test[j]
                #x1 = x_sampled[:,:,-1].clone().detach().long()
                #x2 = x_sampled[:,:,-2].clone().detach().long()
                #x1=self.embedding1(x1)
                #x2=self.embedding2(x2)
                # Ensure both data and model are on the same device (cuda if available)
                if torch.cuda.is_available():
                    x_sampled, y_sampled = x_sampled.cuda(), y_sampled.cuda()
                    self.model.cuda()  # Ensure the model is on the GPU if CUDA is available
                #print(x_sampled.size(),x1.size(),x2.size())
                #x_sampled = torch.cat((x_sampled[:,:,:-2], x1,x2), dim=2)
                y_pred = self.model(x_sampled)
                y_sampled = y_sampled.unsqueeze(0)
                print(y_pred*std+mean, y_sampled*std+mean)
                the_loss = F.mse_loss(y_pred*std+mean, y_sampled*std+mean)
                running_loss += the_loss.item()
                print(the_loss)
        
        return float(running_loss / len(self.train_out_test))
