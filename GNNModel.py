import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from torch_geometric.nn import global_mean_pool as gmeanp, global_max_pool as gmaxp, global_add_pool as gaddp
from torch_geometric.nn import GraphConv

from layers import GCN, HGPSLPool

class GNN(torch.nn.Module):
  def __init__(self, input_dim,hidden_channels,target,num_layers=2 ):
    super(GNN, self).__init__()

    self.conv1 = GraphConv(in_channels=input_dim, out_channels=hidden_channels, aggr='add')
    self.conv2 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels, aggr='add')
    self.conv3 = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels//2, aggr='add')


    self.mlp = Sequential(Linear(hidden_channels//2, hidden_channels//4), ReLU(), Linear(hidden_channels//4, hidden_channels//8), ReLU(), Linear(hidden_channels//8, 1))

    # self.conv1 = GATConv(input_dim, hidden_channels, heads=8, dropout=0.6) 
    # self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, dropout=0.6)
    # self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=8, dropout=0.6)
    # self.lin = torch.nn.Linear(hidden_channels * 8, 1)


    # self.node_emb = BondEncoder(emb_dim=70)

    # aggregators = ['mean', 'min', 'max', 'std']
    # scalers = ['identity', 'amplification', 'attenuation']

    # self.convs = ModuleList()
    # for _ in range(num_layers):
    #   conv = GraphConv(in_channels=hidden_channels, out_channels=70, aggregators=aggregators, scalers=scalers, deg=deg, post_layers=1)
    #   self.convs.append(conv)
    # self.batch_norms = ModuleList()
    # for _ in range(4):
    #   conv = PNAConvSimple(in_channels=70, out_channels=70, aggregators=aggregators, scalers=scalers, deg=deg, post_layers=1)
    #   self.convs.append(conv)
    #   self.batch_norms.append(BatchNorm(70))

    # 
    # self.num_features = input_dim
    # self.nhid = hidden_channels
    # self.num_classes = 1
    # self.pooling_ratio = 0.3
    # self.dropout_ratio = 0.5
    # self.sample = True
    # self.sparse = True
    # self.sl = True
    # self.lamb = 1


    # self.conv1 = GCNConv(self.num_features, self.nhid)
    # self.conv2 = GCN(self.nhid, self.nhid)
    # self.conv3 = GCN(self.nhid, self.nhid)

    # self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
    # self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

    # self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
    # self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
    # self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)


    # MODEL ARCHITECTURE
    # self.node_encoder = Linear(input_dim, hidden_channels)
    # self.edge_encoder = Linear(1, hidden_channels)
    

    # self.layers = torch.nn.ModuleList()
    # for i in range(1, num_layers + 1):
    #   conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
    #                   t=1.0, learn_t=True, num_layers=2, norm='layer')
    #   norm = LayerNorm(hidden_channels, elementwise_affine=True)
    #   act = ReLU(inplace=False)

    #   layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.4,
    #                         ckpt_grad=i % 3)
    #   self.layers.append(layer)



    # MODEL CLASS ATTRIBUTES
    self.target = {'valence':0,'arousal':1,'dominance':2,'liking':3}[target]
    self.best_val_mse = float('inf')
    self.best_epoch = 0
    self.train_losses = []
    self.eval_losses = []
    self.eval_patience_count = 0
    self.eval_patience_reached = False

  def forward(self, x, edge_index, batch, edge_attr):
    
    x = F.dropout(x, p=0.5, training=self.training)

    x = torch.tanh(self.conv1(x, edge_index, edge_attr))
    x = torch.tanh(self.conv2(x, edge_index, edge_attr))
    # x = F.dropout(x, p=0.25, training=self.training)
    x = torch.tanh(self.conv3(x, edge_index, edge_attr))

    # Graph READOUT
    x = gaddp(x, batch)
    x = F.dropout(x, p=0.25, training=self.training)

    x = self.mlp(x)

    return x
    # edge_attr = None

    # x = F.tanh(self.conv1(x, edge_index, edge_attr))
    # x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
    # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

    # x = F.tanh(self.conv2(x, edge_index, edge_attr))
    # x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
    # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

    # x = F.tanh(self.conv3(x, edge_index, edge_attr))
    # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

    # x = F.tanh(x1) + F.tanh(x2) + F.tanh(x3)

    # x = F.tanh(self.lin1(x))
    # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
    # x = F.tanh(self.lin2(x))
    # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
    # x = self.lin3(x)
    # x = self.node_encoder(x)
    # edge_attr = self.edge_encoder(edge_attr.unsqueeze(1))

    # x = self.layers[0].conv(x, edge_index, edge_attr)
    # x = F.dropout(x, p=0.25, training=self.training)
    # for layer in self.layers[1:]:
    #     x = layer(x, edge_index, edge_attr)
    # x = self.layers[0].act(self.layers[0].norm(x))
    # x = global_mean_pool(x, batch)
    # x = F.dropout(x, p=0.25, training=self.training)
    # x = self.mlp(x)
    # return x
    # x = rearrange(x, 'b f -> f b')
    # print(x.shape)
    # x = self.node_emb(x)
    # print(x.shape)
    # for conv, batch_norm in zip(self.convs, self.batch_norms):
    #       h = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
    #       x = h + x  # residual#
    #       x = F.dropout(x, 0.3, training=self.training)
    # x = global_mean_pool(x, batch)
    # return self.mlp(x)
    # x = F.dropout(x, p=0.6, training=self.training)
    # x = self.conv1(x, edge_index)
    # x = F.elu(self.conv1(x, edge_index))
    # x = F.dropout(x, p=0.6, training=self.training)
    # x = self.conv2(x, edge_index)
    
    # Sigmoid and tanh work -> Relu doesnt -> Why?
    # x = torch.relu(x)
    # x = F.dropout(x, p=0.25, training=self.training)
    # x = self.conv2(x, edge_index)
    # x = torch.relu(x)
    # x = F.dropout(x, p=0.25, training=self.training)
    # x = self.conv3(x, edge_index)

    # Graph READOUT
    # x = global_add_pool(x, batch)
  
    # x = F.dropout(x, p=0.25, training=self.training)
    # x = self.lin(x)
    # x = torch.relu(x)

    # return x

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
        self.eval_best_count = 0
      # Early stopping
      elif early_stopping_patience is not None:
          self.eval_patience_count += 1
          if self.eval_patience_count >= early_stopping_patience:
            self.eval_patience_reached = True

    return e_mse, e_l1