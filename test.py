#!/usr/bin/env python

import torch
import numpy as np
import torch.nn.functional as F
from models.GNNLSTM import GNNLSTM
from DEAPDataset import DEAPDataset, train_val_test_split
from torch_geometric.data import DataLoader
import itertools
np.set_printoptions(precision=2)

def test(args):
  ROOT_DIR = './'
  RAW_DIR = 'data/matlabPREPROCESSED'
  PROCESSED_DIR = 'data/graphProcessedData'
  dataset = DEAPDataset(root= ROOT_DIR, raw_dir= RAW_DIR, processed_dir=PROCESSED_DIR, participant_from=args.participant_from, participant_to=args.participant_to)
  # 5 testing samples per participant (30/5/5)
  _, _, test_set = train_val_test_split(dataset)

  test_loader = DataLoader(test_set, batch_size=args.batch_size)

   # MODEL PARAMETERS
  in_channels = test_set[0].num_node_features

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Device: {device}')

  # Instantiate models
  targets = ['valence','arousal','dominance','liking'][:args.n_targets]
  models = [GNNLSTM(in_channels,hidden_channels=64, target=target).to(device).eval() for target in targets]

  # Load best performing params on validation
  for i in range(len(targets)):
    models[i].load_state_dict(torch.load(f'./best_params_{i}'))

  mses = []
  for batch in test_loader:
    batch = batch.to(device)
    predictions = [model(batch, visualize_convolutions=False) for model in models]
    predictions = torch.stack(predictions,dim=1).squeeze()
    print('-Predictions-')
    print(predictions.cpu().detach().numpy(),'\n')
    print('-Ground truth-')
    print(batch.y.cpu().detach().numpy(),'\n')
    mse = F.mse_loss(predictions,batch.y.narrow(1,0,len(targets))).item()
    mses.append(mse)
    print(f'Mean average error: {F.l1_loss(predictions,batch.y.narrow(1,0,len(targets))).item()}')
    print(f'Mean squared error: {mse}')

  print('----------------')
  print(f'MEAN SQUARED ERROR FOR TEST SET: {np.array(mses).mean()}')