'''
code for DP-SGD and 
running regular SGD on locally private data
'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Flatten the input if needed
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class ProbitNLLLoss(nn.Module):
    def __init__(self):
        super(ProbitNLLLoss, self).__init__()

    def forward(self, inputs, targets):
        # Calculate the cumulative distribution function (CDF) of a standard normal distribution
        cdf = 0.5 * (1 + torch.erf(inputs / (2**0.5)))

        # Calculate the negative log-likelihood of the Probit Model
        loss = -torch.log(cdf.clamp(min=1e-10)) * targets - torch.log((1 - cdf).clamp(min=1e-10)) * (1 - targets)
        return loss.mean()


def train(model, train_loader, epoch, optimizer, delta, privacy_engine, device):
    model.train()
    criterion = ProbitNLLLoss()
    losses = []
    for batch in train_loader:
        data = batch['data']
        target = batch['label']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.unsqueeze(1)
        loss = criterion(output, target)
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
    criterion = ProbitNLLLoss()
    test_loss = 0

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


def dp_sgd(train_loader, test_loader, eps, delta, central):
    max_epochs = 2
    max_grad_norm = 1.0
    
    model = Net(input_dim=5, hidden_dim=2, output_dim=1).to(device="cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    privacy_engine = None
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
    return top_acc