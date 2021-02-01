#!/usr/bin/env python

import torch
from torch_geometric.data import DataLoader
from DEAPDataset import DEAPDataset, train_val_test_split
from GNNModel import GNN


ROOT_DIR = './'
RAW_DIR = 'data/matlabPREPROCESSED'
PROCESSED_DIR = 'data/graphProcessedData'

dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR, participant_from=1, participant_to=32)

train_set, val_set, _ = train_val_test_split(dataset)

BATCH_SIZE = 8
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

EPOCH_N = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Define loss function 
criterion = torch.nn.MSELoss()

# Instantiate models
in_channels = train_set[0].num_node_features
targets = ['valence','arousal','dominance','liking']
models = [GNN(in_channels,hidden_channels=64, target=target).to(device) for target in targets]

# Instantiate optimizers
optimizers = [torch.optim.Adam(model.parameters()) for model in models]

for epoch in range(EPOCH_N):
  # Train epoch for every model
  t_e_losses = [model.train_epoch(train_loader,optimizers[i],criterion,device) for i,model in enumerate(models)]

  # Validation epoch for every model
  v_e_losses = [model.eval_model(val_loader,device,epoch,val=True) for model in models]

  # Epoch results
  print(f'------ Epoch {epoch} ------ \n')
  for i,target in enumerate(targets):
    print(f'{target}: Train loss: {t_e_losses[i]:.2f} | Validation mse: {v_e_losses[i][0]:.2f}')

# Picking best performing model on validation
for i in range(4):
  models[i].load_state_dict(torch.load(f'./best_params_{i}'))

# Evaluating best models
final_eval = [model.eval_model(val_loader,device,val=False) for model in models]
print(f'------ Final model eval ------ \n')
for i,target in enumerate(targets):
  print(f'{target} (epoch {models[i].best_epoch}): Validation mse: {final_eval[i][0]:.2f}')
