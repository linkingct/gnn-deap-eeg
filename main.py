#!/usr/bin/env python

import torch
from torch_geometric.data import DataLoader
from DEAPDataset import DEAPDataset, plot_graph
from GNNModel import GNN

ROOT_DIR = './'
RAW_DIR = 'data/matlabPREPROCESSED'
PROCESSED_DIR = 'data/graphProcessedData'

dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR, participant=0)

train_dataset = dataset[0:30]
val_dataset = dataset[30:35]
test_dataset = dataset[35:40]

data = train_dataset[0]  # Get the first graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

EPOCH_N = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Instantiate models
in_channels = train_dataset.num_node_features
valence_model = GNN(in_channels,hidden_channels=64, target='valence').to(device)
arousal_model = GNN(in_channels,hidden_channels=64, target='arousal').to(device)
dominance_model = GNN(in_channels,hidden_channels=64, target='dominance').to(device)
liking_model = GNN(in_channels,hidden_channels=64, target='liking').to(device)

criterion = torch.nn.MSELoss()

# Instantiate optimizers
valence_optimizer = torch.optim.Adam(valence_model.parameters())
arousal_optimizer = torch.optim.Adam(arousal_model.parameters())
dominance_optimizer = torch.optim.Adam(dominance_model.parameters())
liking_optimizer = torch.optim.Adam(liking_model.parameters())

for epoch in range(EPOCH_N):
  # Train epoch for every model
  t_e_loss_valence = valence_model.train_epoch(train_loader,valence_optimizer,criterion,device)
  t_e_loss_arousal = arousal_model.train_epoch(train_loader,arousal_optimizer,criterion,device)
  t_e_loss_dominance = dominance_model.train_epoch(train_loader,dominance_optimizer,criterion,device)
  t_e_loss_liking = liking_model.train_epoch(train_loader,liking_optimizer,criterion,device)

  # Validation epoch for every model
  v_e_mse_valence, v_e_l1_valence = valence_model.eval_model(val_loader,device,epoch,val=True)
  v_e_mse_arousal, v_e_l1_arousal = arousal_model.eval_model(val_loader,device,epoch,val=True)
  v_e_mse_dominance, v_e_l1_dominance = dominance_model.eval_model(val_loader,device,epoch,val=True)
  v_e_mse_liking, v_e_l1_liking = liking_model.eval_model(val_loader,device,epoch,val=True)

  # Epoch results
  print(f'------ Epoch {epoch} ------ \n')
  print(f'Valence: Train loss: {t_e_loss_valence:.2f} | Validation mse: {v_e_mse_valence:.2f} | Validation l1: {v_e_l1_valence:.2f}')
  print(f'Arousal: Train loss: {t_e_loss_arousal:.2f} | Validation mse: {v_e_mse_arousal:.2f} | Validation l1: {v_e_l1_arousal:.2f}')
  print(f'Dominance: Train loss: {t_e_loss_dominance:.2f} | Validation mse: {v_e_mse_dominance:.2f} | Validation l1: {v_e_l1_dominance:.2f}')
  print(f'Liking: Train loss: {t_e_loss_liking:.2f} | Validation mse: {v_e_l1_liking:.2f} | Validation l1: {v_e_l1_liking:.2f}')

# Picking best performing model on validation
valence_model.load_state_dict(torch.load(f'./best_params_{valence_model.target}'))
arousal_model.load_state_dict(torch.load(f'./best_params_{arousal_model.target}'))
dominance_model.load_state_dict(torch.load(f'./best_params_{dominance_model.target}'))
liking_model.load_state_dict(torch.load(f'./best_params_{liking_model.target}'))

# Evaluating best models
v_e_mse_valence, v_e_l1_valence = valence_model.eval_model(val_loader,device,val=False)
v_e_mse_arousal, v_e_l1_arousal = arousal_model.eval_model(val_loader,device,val=False)
v_e_mse_dominance, v_e_l1_dominance = dominance_model.eval_model(val_loader,device,val=False)
v_e_mse_liking, v_e_l1_liking = liking_model.eval_model(val_loader,device,val=False)

print(f'------ Final model eval ------ \n')
print(f'Valence (epoch {valence_model.best_epoch}): Validation mse: {v_e_mse_valence:.2f} | Validation l1: {v_e_l1_valence:.2f}')
print(f'Arousal (epoch {arousal_model.best_epoch}): Validation mse: {v_e_mse_arousal:.2f} | Validation l1: {v_e_l1_arousal:.2f}')
print(f'Dominance (epoch {dominance_model.best_epoch}): Validation mse: {v_e_mse_dominance:.2f} | Validation l1: {v_e_l1_dominance:.2f}')
print(f'Liking (epoch {liking_model.best_epoch}): Validation mse: {v_e_l1_liking:.2f} | Validation l1: {v_e_l1_liking:.2f}')