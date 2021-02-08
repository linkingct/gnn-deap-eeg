import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from torch_geometric.nn import global_mean_pool as gmeanp, global_max_pool as gmaxp, global_add_pool as gaddp
from torch_geometric.nn import GraphConv

from einops import reduce

from layers import GCN, HGPSLPool

class GNN(torch.nn.Module):
  def __init__(self, input_dim,hidden_channels,target,num_layers=2 ):
    super(GNN, self).__init__()

    self.gconv1 = GraphConv(in_channels=input_dim, out_channels=hidden_channels*2, aggr='add')
    self.gconv2 = GraphConv(in_channels=hidden_channels*2, out_channels=hidden_channels, aggr='add')
    self.gconv3 = GraphConv(in_channels=hidden_channels, out_channels=64, aggr='add')

    self.conv1 = nn.Conv1d(1, 1, 3, 2)
    self.conv2 = nn.Conv1d(1, 1, 3, 2)
    self.conv3 = nn.Conv1d(1, 1, 5, 2)
    self.conv4 = nn.Conv1d(1, 1, 6)


    self.mlp = Sequential(Linear(32, 16), ReLU(), Linear(16, 8), ReLU(), Linear(8, 1))

    # MODEL CLASS ATTRIBUTES
    self.target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target]
    self.best_val_mse = float('inf')
    self.best_epoch = 0
    self.train_losses = []
    self.eval_losses = []
    self.eval_patience_count = 0
    self.eval_patience_reached = False

  def forward(self, x, edge_index, batch, edge_attr):
    
    # Graph convs
    x = F.dropout(x, p=0.5, training=self.training)
    x = torch.tanh(self.gconv1(x, edge_index, edge_attr))
    x = torch.tanh(self.gconv2(x, edge_index, edge_attr))
    x = F.dropout(x, p=0.25, training=self.training)
    x = torch.tanh(self.gconv3(x, edge_index, edge_attr))

    # 1d convs
    x = torch.unsqueeze(x,1)
    x = torch.relu(self.conv1(x))
    x = F.dropout(x, p=0.25, training=self.training)
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    x = torch.relu(self.conv4(x))
    x = reduce(x,'(b x) c d -> b x','mean',x=32)

    
   
    # Graph READOUT
    # x = gaddp(x, batch)

    
    x = F.dropout(x, p=0.25, training=self.training)
    # mlp
    x = self.mlp(x)

    return x
    

  def train_epoch(self,loader,optim,criterion,device):
    if self.eval_patience_reached:
      return -1
    self.train()
    epoch_losses = []
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

      loss = mse_loss 
      # loss = mse_loss
      loss.backward()
      optim.step()
      epoch_losses.append(mse_loss.item())
    epoch_mean_loss = np.array(epoch_losses).mean()
    self.train_losses.append(epoch_mean_loss)
    return epoch_mean_loss

  def eval_model(self,loader,device,epoch=-1, model_is_training = False, early_stopping_patience = None):
    if self.eval_patience_reached and model_is_training:
      return -1,-1
    self.eval()
    mses = []
    l1s = []
    for batch in loader:
      batch = batch.to(device)
      out = self(batch.x.float(),batch.edge_index,batch.batch,batch.edge_attr.float())
      target = batch.y.T[self.target].unsqueeze(1)
      mses.append(F.mse_loss(out,target).item())
      l1s.append(F.l1_loss(out,target).item())
    e_mse, e_l1 = np.array(mses).mean(), np.array(l1s).mean()

    if model_is_training:
      self.eval_losses.append(e_mse)
      # Save current best model locally
      if e_mse < self.best_val_mse:
        self.best_val_mse = e_mse
        self.best_epoch = epoch
        torch.save(self.state_dict(),f'./best_params_{self.target}')
        self.eval_patience_count = 0
      # Early stopping
      elif early_stopping_patience is not None:
          self.eval_patience_count += 1
          if self.eval_patience_count >= early_stopping_patience:
            self.eval_patience_reached = True

    return e_mse, e_l1