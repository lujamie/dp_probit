'''
code for DP-SGD and 
running regular SGD on locally private data
'''

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.clone().detach().float()
        self.labels = labels.clone().detach().float()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
        return sample


def torch_data(X, y):
    bat_size = 1
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    TD = CustomDataset(X_tensor, y_tensor)
    data_loader = DataLoader(TD, batch_size=bat_size, shuffle=True)
    return data_loader

class Probit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Probit, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        z = self.linear(x)
        # rho = np.random.standard_normal()
        # output = norm.cdf((xbeta+rho).detach().numpy())
        #return torch.tensor(output).float()
        return torch.sigmoid(z)

class ProbitNLLLoss(nn.Module):
    def __init__(self, weights):
        super(ProbitNLLLoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        # Calculate the cumulative distribution function (CDF) of a standard normal distribution
        z = torch.mul(inputs, self.weights)
        cdf = (1/math.sqrt(2*math.pi)) * torch.exp(-(z**2)/2)

        # Calculate the negative log-likelihood of the Probit Model
        loss = -torch.log(cdf.clamp(min=1e-10)) * targets - torch.log((1 - cdf).clamp(min=1e-10)) * (1 - targets)
        return loss.mean()


def train(model, train_loader, epoch, optimizer, delta, privacy_engine, device):
    model.train()
    #criterion = ProbitNLLLoss(weights=model.linear.weight)
    criterion = nn.BCELoss()
    losses = []
    for batch in train_loader:
        data = batch['data']
        target = batch['label']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.unsqueeze(1)
        #print(f"output={output}, target={target}")
        loss = criterion(output, target)
        #loss.requires_grad = True
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {delta})"
        )

def test(model, test_loader, device):
    model.eval()
    #criterion = ProbitNLLLoss(weights=model.linear.weight)
    criterion = nn.BCELoss()
    test_loss = 0
    correct = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch['data']
            target = batch['label']
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.unsqueeze(1)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = (output > 0.5).float()
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(
            test_loss,
            100.0 * accuracy,
        )
    )
    return accuracy


def dp_sgd(train_loader, test_loader, eps, delta, K, central):
    max_epochs = 5
    max_grad_norm = 1.0
    
    model = Probit(input_dim=K, output_dim=1).to(device="cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    if central:
        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=max_epochs,
            target_epsilon=eps,
            target_delta=delta,
            max_grad_norm=1.0,
        )
        print(f"Using sigma={optimizer.noise_multiplier} and C={max_grad_norm}")
        for epoch in range(max_epochs):
            train(model, train_loader, epoch, optimizer, delta, privacy_engine, device="cpu")
        top_acc = test(model, test_loader, device="cpu")
    else:
        privacy_engine = None

        # for epoch in range(max_epochs):
        #     train(model, train_loader, epoch, optimizer, delta, privacy_engine, device="cpu")
        # top_acc = test(model, test_loader, device="cpu")

        
        model_dict = model.state_dict()
        n_iter = math.ceil(math.sqrt(len(train_loader.dataset)))
        for i in range(n_iter):
            for epoch in range(max_epochs):
                train(model, train_loader, epoch, optimizer, delta, privacy_engine, device="cpu")
            new_model_dict = model.state_dict()
            for key in model_dict:
                model_dict[key] += new_model_dict[key]
        for key in model_dict:
            model_dict[key] /= n_iter
        avg_model = Probit(input_dim=K, output_dim=1).to(device="cpu")
        avg_model.load_state_dict(model_dict)
    
        top_acc = test(avg_model, test_loader, device="cpu")
    return top_acc