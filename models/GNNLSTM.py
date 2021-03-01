import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from torch_geometric.nn import global_mean_pool as gmeanp, global_max_pool as gmaxp, global_add_pool as gaddp
from torch_geometric.nn import GraphConv

from einops import reduce, rearrange

# from layers import GCN, HGPSLPoo
from DEAPDataset import visualize_graph

class GNNLSTM(torch.nn.Module):
  def __init__(self, input_dim,hidden_channels,target,num_layers=2 ):
    super(GNNLSTM, self).__init__()

    self.gconv1 = GraphConv(in_channels=8064, out_channels=5000, aggr='add')
    self.gconv2 = GraphConv(in_channels=5000, out_channels=4000, aggr='add')

    self.lstm = nn.LSTM(2, 3, 2,bidirectional=True)

    self.mlp = Sequential(Linear(15000, 1))

    # MODEL CLASS ATTRIBUTES
    self.target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target]
    self.best_val_mse = float('inf')
    self.best_epoch = 0
    self.train_losses = []
    self.eval_losses = []
    self.eval_patience_count = 0
    self.eval_patience_reached = False

  def forward(self, batch, visualize_convolutions = False):
    x = batch.x
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    batch = batch.batch
    bs = len(torch.unique(batch))
    # Information propagation trough graph visualization
    if visualize_convolutions:
      visualize_graph(x[:32])
    
    x = self.gconv1(x,edge_index,edge_attr)
    x = F.dropout(x, p=0.3, training=self.training)

    if visualize_convolutions:
      visualize_graph(x[:32])

    # x = self.gconv2(x,edge_index,edge_attr)
    # x = F.dropout(x, p=0.2, training=self.training)

    # if visualize_convolutions:
    #   visualize_graph(x[:32])
    x = gaddp(x, batch)
    # print(x.shape)
    x = rearrange(x,'b (sl i) -> sl b i',i=2)
    # print(x.shape)
    output, (c_n,h_n)  = self.lstm(x)
    # print(x.shape)
    x = rearrange(output,'sl b i -> b (sl i)')
    x = self.mlp(x)

    return x