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

    self.gconv1 = GraphConv(in_channels=672, out_channels=400, aggr='add')
    self.gconv2 = GraphConv(in_channels=400, out_channels=300, aggr='add')
    # self.gconv3 = GraphConv(in_channels=250, out_channels=64, aggr='add')
    # self.gconv3 = GraphConv(in_channels=(12,672), out_channels=64, aggr='add')

    self.conv1 = nn.Conv1d(12, 8, 5, 2)
    self.conv2 = nn.Conv1d(8, 4, 5, 2)
    self.conv3 = nn.Conv1d(4, 1, 5, 1)
    # self.conv4 = nn.Conv1d(1, 1, 6)


    self.mlp = Sequential(Linear(68, 32), ReLU(), Linear(32, 16), ReLU(), Linear(16, 1))

    # MODEL CLASS ATTRIBUTES
    self.target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target]
    self.best_val_mse = float('inf')
    self.best_epoch = 0
    self.train_losses = []
    self.eval_losses = []
    self.eval_patience_count = 0
    self.eval_patience_reached = False

  def forward(self, batch, visualize_convolutions = False):
    x = batch.x.float()
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr.float()
    bs = len(torch.unique(batch.batch))

    # Information propagation trough graph visualization
    if visualize_convolutions:
      visualize_window(x)
    
    # GRAPH CONVOLUTIONS (SPATIAL)
    x = rearrange(x,'w bs c ->(w bs) c')
    x = torch.tanh(self.gconv1(x, edge_index, edge_attr))
    x = F.dropout(x, p=0.25, training=self.training)

    # if visualize_convolutions:
    #   visualize_window(rearrange(x,'(w bs) c -> w bs c', bs=32))
       
    x = torch.tanh(self.gconv2(x, edge_index, edge_attr))
    x = F.dropout(x, p=0.25, training=self.training)

    # if visualize_convolutions:
    #   visualize_window(rearrange(x,'(w bs) c -> w bs c', bs=32))

    # x = torch.tanh(self.gconv3(x, edge_index, edge_attr))
    # x = F.dropout(x, p=0.3, training=self.training)
    
    if visualize_convolutions:
      visualize_window(rearrange(x,'(w bs) c -> w bs c', bs=32))

    # print(x.shape)

    # READOUT (POOLING) FUNCTION
    x = gaddp(x, torch.tensor(np.repeat(np.array(range(0,12*bs)),32)).to('cuda'))

    # print(x.shape)
    
    # 1D CONVS (TEMPORAL)
    x = rearrange(x,'(bs g) f -> bs g f', bs=bs)

    # print(x.shape)

    x = torch.relu(self.conv1(x))
    x = F.dropout(x, p=0.1, training=self.training)
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    x = rearrange(x,'bs o e -> bs (o e)', bs=bs)

    # print(x.shape)

    # FINAL REGRESSION
    x = self.mlp(x)

    return x