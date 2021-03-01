#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from DEAPDataset import DEAPDataset, train_val_test_split, plot_graph, describe_graph, plot_graph
from models.GNNLSTM import GNNLSTM
from matplotlib import pyplot as plt
from tqdm import tqdm

# Define training and eval functions for each epoch (will be shared for all models)
def train_epoch(model,loader,optim,criterion,device):
    if model.eval_patience_reached:
      return -1
    model.train()
    epoch_losses = []
    for batch in tqdm(loader):
      optim.zero_grad()
      batch = batch.to(device)
      out = model(batch, visualize_convolutions=False)
      # Gets first label for every graph
      target = batch.y.T[model.target].unsqueeze(1)
      mse_loss = criterion(out, target)
      # REGULARIZATION
      l1_regularization, l2_regularization = torch.tensor(0, dtype=torch.float).to(device), torch.tensor(0, dtype=torch.float).to(device)
      for param in model.parameters():
        l1_regularization += (torch.norm(param, 1)**2).float()
        # l2_regularization += (torch.norm(param, 2)**2).float()
      loss = mse_loss + 0.02 * l1_regularization
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


def train (args):
  # TODO: set as args
  ROOT_DIR = './'
  RAW_DIR = 'data/matlabPREPROCESSED'
  PROCESSED_DIR = 'data/graphProcessedData'
  # Initialize dataset
  dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR, participant_from=args.participant_from, participant_to=args.participant_to)
  # 30 samples are used for training, 5 for validation and 5 are saved for testing
  train_set, val_set, _ = train_val_test_split(dataset)
  # Describe graph structure (same for all instances)
  describe_graph(train_set[0])
  # Set batch size
  BATCH_SIZE = args.batch_size
  train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=not args.dont_shuffle_train)
  val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
  # Use GPU if available
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Device: {device}')
  # Define loss function 
  criterion = torch.nn.MSELoss()
  # Define model targets. Each target has a model associated to it.
  targets = ['valence','arousal','dominance','liking'][:args.n_targets]

  # MODEL PARAMETERS
  in_channels = train_set[0].num_node_features

  # Print losses over time (train and val)
  plt.figure(figsize=(10, 10))
  # Train models one by one as opposed to having an array [] of models. Avoids CUDA out of memory error
  MAX_EPOCH_N = args.max_epoch
  for i,target in enumerate(targets):
    print(f'Now training {target} model')
    model = GNNLSTM(in_channels,hidden_channels=64, target=target).to(device)
    optim = torch.optim.Adam(model.parameters())
    for epoch in range(MAX_EPOCH_N):
      # Train epoch for every model
      t_e_loss = train_epoch(model,train_loader,optim,criterion,device)
      # Validation epoch for every model
      v_e_loss = eval_epoch(model,val_loader,device,epoch,model_is_training = True, early_stopping_patience=4) 
      # Break if model has reached patience limit. Model parameters are saved to 'best_params' file.
      if t_e_loss == -1:
        break
      # Epoch results
      print(f'------ Epoch {epoch} ------')
      print(f'{target}: Train e. mse: {t_e_loss:.2f} | Validation e. mse: {v_e_loss[0]:.2f}')
    plt.subplot(2,2,i+1)
    plt.plot(model.train_losses)
    plt.plot(model.eval_losses)
    plt.title(f'{target} losses')
    plt.ylabel('loss (mse)')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
#   # models[i].load_state_dict(torch.load(f'./best_params_{i}'))
  plt.savefig('train_losses.png')

  # Print losses over time (train and val)
  # plt.figure(figsize=(10, 10))
  # Load best performing parameters on validation for each model
  print(f'------ Final model eval ------ \n')
  for i,target in enumerate(targets):
    model = GNNLSTM(in_channels,hidden_channels=64, target=target).to(device)
    model.load_state_dict(torch.load(f'./best_params_{i}'))
    # Evaluating best models
    final_eval = eval_epoch(model, val_loader,device,model_is_training = False)
    print (f'{target} (epoch {model.best_epoch}): Validation mse: {final_eval[0]:.2f}')



# for i,target in enumerate(targets):

# 
