import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

class GNN(torch.nn.Module):
  def __init__(self, input_dim,hidden_channels, target='valence'):
    super(GNN, self).__init__()
    self.conv1 = GraphConv(input_dim, hidden_channels)  
    self.conv2 = GraphConv(hidden_channels, hidden_channels)
    self.conv3 = GraphConv(hidden_channels, hidden_channels)
    self.lin = torch.nn.Linear(hidden_channels, 1)
    self.target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target]
    self.best_val_mse = float('inf')
    self.best_epoch = 0

  def forward(self, x, edge_index, batch,edge_attr):
    x = self.conv1(x, edge_index,edge_attr)
    # Sigmoid and tanh work -> Relu doesnt -> Why?
    x = torch.relu(x)
    x = F.dropout(x, p=0.5, training=self.training)
    # x = self.conv2(x, edge_index,edge_attr)
    # x = torch.tanh(x)
    # x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv3(x, edge_index,edge_attr)

    # Graph READOUT
    x = global_mean_pool(x, batch)
  
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.lin(x)
    x = torch.relu(x)

    return x

  def train_epoch(self,loader,optim,criterion,device):
    self.train()
    losses = []
    for batch in loader:
      batch = batch.to(device)
      optim.zero_grad()
      out = self(batch.x.float(),batch.edge_index,batch.batch,batch.edge_attr.float())
      # Gets first label for every graph
      target = batch.y.T[self.target].unsqueeze(1)
      mse_loss = criterion(out, target)

      # REGULARIZATION
      l1_regularization, l2_regularization = torch.tensor(0, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device)
      for param in self.parameters():
        l1_regularization += (torch.norm(param, 1)**2).float()
        l2_regularization += (torch.norm(param, 2)**2).float()

      loss = mse_loss + 0.05 * l1_regularization 
      # loss = mse_loss
      loss.backward()
      optim.step()
      losses.append(mse_loss.item())
    return np.array(losses).mean()

  def eval_model(self,loader,device,epoch=-1,print_predictions=False, val=False):
    self.eval()
    mses = []
    l1s = []
    for batch in loader:
      batch = batch.to(device)
      out = self(batch.x.float(),batch.edge_index,batch.batch,batch.edge_attr.float())
      target = batch.y.T[self.target].unsqueeze(1)
      if print_predictions:
        print(f'Predictions:\n {out.cpu().detach().numpy()} \n Ground truth:\n {target.cpu().detach().numpy()}')
      mses.append(F.mse_loss(out,target).item())
      l1s.append(F.l1_loss(out,target).item())

    e_mse, e_l1 = np.array(mses).mean(), np.array(l1s).mean()

    # and abs(t_e_loss - v_e_mse) < 5
    if val == True and e_mse < self.best_val_mse :
      self.best_val_mse = e_mse
      self.best_epoch = epoch
      torch.save(self.state_dict(),f'./best_params_{self.target}')
    return e_mse, e_l1