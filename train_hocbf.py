import os
import numpy as np
import torch
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GAT
from torch_geometric.data import DataLoader
from qpth.qp import QPFunction

from dataset.road_scene_graph import HeteroRoadSceneDataset



class Model(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.gat = GAT(in_channels, hidden_channels, num_layers, 1)


    def forward(self, x, edge_index, q, G, h, lb, ub, batch):
        h = self.gat(x, edge_index)
        P = torch.diag(torch.cat([1.0, h],))
        e = torch.autograd.Variable(torch.Tensor())
        n = lb.shape[0]
        u = QPFunction(verbose=False)(P, q, torch.cat([G, torch.eye(n), torch.eye(n)], 0), torch.cat([h, torch.eye(lb), torch.eye(lb)], 0), e, e)
        return u

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

train_dataset = HeteroRoadSceneDataset(os.path.join('dataset', 'rollout_hocbf'), episode=1)
train_loader = DataLoader(train_dataset)

model = Model(train_dataset.num_features, 128, train_dataset.num_classes, concat=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = torch.nn.MSELoss()
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = total_examples = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += data.num_graphs

    return total_correct / total_examples




if __name__=="__main__":
    train()
    for epoch in range(1, 61):
        loss = train()
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')