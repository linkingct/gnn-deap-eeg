#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from DEAPDataset import DEAPDataset, train_val_test_split, plot_graph
from GNNModel import GNN
from matplotlib import pyplot as plt


ROOT_DIR = './'
RAW_DIR = 'data/matlabPREPROCESSED'
PROCESSED_DIR = 'data/graphProcessedData'

dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR, participant_from=1, participant_to=32, window_size=672)

train_set, val_set, _ = train_val_test_split(dataset)

graph = train_set[0]

BATCH_SIZE = 2
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

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

# Define training and eval functions for each epoch (will be shared for all models)
def train_epoch(model,loader,optim,criterion,device):
    if model.eval_patience_reached:
      return -1
    model.train()
    epoch_losses = []
    for batch in loader:
      optim.zero_grad()
      batch = batch.to(device)
      out = model(batch)
      # Gets first label for every graph
      target = batch.y.T[model.target].unsqueeze(1)
      mse_loss = criterion(out, target)

      # REGULARIZATION
      l1_regularization, l2_regularization = torch.tensor(0, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device)
      for param in model.parameters():
        l1_regularization += (torch.norm(param, 1)**2).float()
        l2_regularization += (torch.norm(param, 2)**2).float()

      loss = mse_loss + l1_regularization * 0.02 + l2_regularization*0.01
      loss.backward()
      optim.step()
      epoch_losses.append(mse_loss.item())
    epoch_mean_loss = np.array(epoch_losses).mean()
    model.train_losses.append(epoch_mean_loss)
    return epoch_mean_loss

def eval_epoch(model,loader,device,epoch=-1, model_is_training = False, early_stopping_patience = None):
    if model.eval_patience_reached and model_is_training:
      return [-1,-1]
    model.eval()
    mses = []
    l1s = []
    # Evaluation
    for batch in loader:
      batch = batch.to(device)
      out = model(batch)
      target = batch.y.T[model.target].unsqueeze(1)
      mses.append(F.mse_loss(out,target).item())
      l1s.append(F.l1_loss(out,target).item())
    e_mse, e_l1 = np.array(mses).mean(), np.array(l1s).mean()

    # Early stopping and checkpoint
    if model_is_training:
      model.eval_losses.append(e_mse)
      # Save current best model locally
      if e_mse < model.best_val_mse:
        model.best_val_mse = e_mse
        model.best_epoch = epoch
        torch.save(model.state_dict(),f'./best_params_{model.target}')
        model.eval_patience_count = 0
      # Early stopping
      elif early_stopping_patience is not None:
          model.eval_patience_count += 1
          if model.eval_patience_count >= early_stopping_patience:
            model.eval_patience_reached = True

    return e_mse, e_l1

MAX_EPOCH_N = 1000
for epoch in range(MAX_EPOCH_N):
  # Train epoch for every model
  t_e_losses = [train_epoch(model,train_loader,optimizers[i],criterion,device) for i,model in enumerate(models)]
  # Validation epoch for every model
  v_e_losses = [eval_epoch(model,val_loader,device,epoch,model_is_training = True, early_stopping_patience=20) for model in models]

  # Break if all models finished training
  if t_e_losses.count(-1) == len(t_e_losses):
    break

  # Epoch results
  print(f'------ Epoch {epoch} ------ \n')
  for i,target in enumerate(targets):
    if t_e_losses[i] != -1:
      print(f'{target}: Train loss: {t_e_losses[i]:.2f} | Validation mse: {v_e_losses[i][0]:.2f}')

# Picking best performing model on validation
for i in range(4):
  models[i].load_state_dict(torch.load(f'./best_params_{i}'))

# Evaluating best models
final_eval = [model.eval_model(val_loader,device,model_is_training = False) for model in models]
print(f'------ Final model eval ------ \n')
for i,target in enumerate(targets):
  print(f'{target} (epoch {models[i].best_epoch}): Validation mse: {final_eval[i][0]:.2f}')

# Print losses over time (train and val)
plt.figure(figsize=(10, 10))
for i,target in enumerate(targets):
  plt.subplot(2,2,i+1)
  plt.plot(models[i].train_losses)
  plt.plot(models[i].eval_losses)
  plt.title(f'{target} losses')
  plt.ylabel('loss (mse)')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper right')
  # models[i].load_state_dict(torch.load(f'./best_params_{i}'))
plt.savefig('train_losses.png')
