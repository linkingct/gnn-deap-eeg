import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from torch_geometric.nn import global_mean_pool as gmeanp, global_max_pool as gmaxp, global_add_pool as gaddp
from torch_geometric.nn import GraphConv

from einops import reduce, rearrange

# from layers import GCN, HGPSLPoo
from DEAPDataset import visualize_window

class GNN(torch.nn.Module):
  def __init__(self, input_dim,hidden_channels,target,num_layers=2 ):
    super(GNN, self).__init__()

    self.gconv1 = GraphConv(in_channels=672, out_channels=128, aggr='add')
    self.gconv2 = GraphConv(in_channels=128, out_channels=64, aggr='add')
    # self.gconv3 = GraphConv(in_channels=(12,672), out_channels=64, aggr='add')

    self.conv1 = nn.Conv1d(12, 8, 2, 1)
    self.conv2 = nn.Conv1d(8, 4, 2, 1)
    self.conv3 = nn.Conv1d(4, 1, 2, 1)
    # self.conv4 = nn.Conv1d(1, 1, 6)


    self.mlp = Sequential(Linear(61, 16), ReLU(), Linear(16, 8), ReLU(), Linear(8, 1))

    # MODEL CLASS ATTRIBUTES
    self.target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target]
    self.best_val_mse = float('inf')
    self.best_epoch = 0
    self.train_losses = []
    self.eval_losses = []
    self.eval_patience_count = 0
    self.eval_patience_reached = False

  def forward(self, x, edge_index, batch, edge_attr):
    # visualize_window(x)
    bs = len(torch.unique(batch))
    
    # GRAPH CONVOLUTIONS (SPATIAL)
    x = rearrange(x,'w bs c ->(w bs) c')
    x = torch.tanh(self.gconv1(x, edge_index, edge_attr))
    x = F.dropout(x, p=0.25, training=self.training)
    # visualize_window(rearrange(x,'(w bs) c -> w bs c', bs=32) )
    x = torch.tanh(self.gconv2(x, edge_index, edge_attr))
    # visualize_window(x,'(w bs) c -> w bs c', bs=32)

    # READOUT (POOLING) FUNCTION
    x = gmeanp(x, torch.tensor(np.repeat(np.array(range(0,12*bs)),32)).to('cuda'))
    
    # 1D CONVS (TEMPORAL)
    x = rearrange(x,'(bs g) f -> bs g f', bs=bs)
    x = torch.relu(self.conv1(x))
    x = F.dropout(x, p=0.25, training=self.training)
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    x = rearrange(x,'bs o e -> bs (o e)', bs=bs)

    # FINAL REGRESSION
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