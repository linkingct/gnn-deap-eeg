#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from GNNModel import GNN
from DEAPDataset import DEAPDataset, train_val_test_split
from torch_geometric.data import DataLoader
import itertools

np.set_printoptions(precision=2)

ROOT_DIR = './'
RAW_DIR = 'data/matlabPREPROCESSED'
PROCESSED_DIR = 'data/graphProcessedData'

dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR, participant_from=1, participant_to=1)

_, _, test_set = train_val_test_split(dataset)

test_loader = DataLoader(test_set, batch_size=8)

in_channels = test_set[0].num_node_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Instantiate models
targets = ['valence','arousal','dominance','liking']
models = [GNN(in_channels,hidden_channels=64, target=target).to(device).eval() for target in targets]

# Load best performing params on validation
for i in range(4):
  models[i].load_state_dict(torch.load(f'./best_params_{i}'))

for batch in test_loader:
  batch = batch.to(device)
  predictions = [model(batch.x.float(),batch.edge_index,batch.batch,batch.edge_attr.float()) for model in models]
  predictions = torch.stack(predictions,dim=1).squeeze()
  print('-Predictions-')
  print(predictions.cpu().detach().numpy(),'\n')
  print('-Ground truth-')
  print(batch.y.cpu().detach().numpy(),'\n')



  print(f'Mean average error: {F.l1_loss(predictions,batch.y).item()}')
  print(f'Mean squared error: {F.mse_loss(predictions,batch.y).item()}')
