import os
import torch
import scipy
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import mne
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from tqdm import tqdm
from Electrodes import Electrodes
from einops import rearrange


# Get 30 videos for each participant for test, 5 for validation and 5 for testing
def train_val_test_split(dataset):
  train_mask = np.append(np.repeat(1,30),np.repeat(0,10))
  train_mask = np.tile(train_mask,int(len(dataset)/40))
  val_mask = np.append(np.append(np.repeat(0,30),np.repeat(1,5)),np.repeat(0,5))
  val_mask = np.tile(val_mask,int(len(dataset)/40))
  test_mask = np.append(np.repeat(0,35),np.repeat(1,5))
  test_mask = np.tile(test_mask,int(len(dataset)/40))

  train_set = [c for c in itertools.compress(dataset,train_mask)]
  val_set = [c for c in itertools.compress(dataset,val_mask)]
  test_set = [c for c in itertools.compress(dataset,test_mask)]

  return train_set, val_set, test_set

def plot_graph(graph_data):
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx
    from matplotlib import pyplot as plt
    graph = to_networkx(graph_data)

    plt.figure(1,figsize=(7,6)) 
    nx.draw(graph, cmap=plt.get_cmap('Set1'),node_size=75,linewidths=6)
    plt.show()

def visualize_window(window):
  window = window.cpu().detach().numpy()[:12]

  eeg_mean = window.mean(axis=-1)
  chunks = eeg_mean.T
  electrodes = Electrodes()
  # Show all chunks and label (12)
  fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30,20), gridspec_kw = {'wspace':0, 'hspace':0.2})
  for i,chunk in enumerate(eeg_mean):
      index = np.unravel_index(i, (3,4))
      ax = axes[index[0]][index[1]]
      ax.title.set_text(f'Chunk number {i} (seconds {5.25*i} to {5.25*(i+1)})')
      im,_ = mne.viz.topomap.plot_topomap(chunk,electrodes.positions_2d,names=electrodes.channel_names,show_names=True,axes=ax,cmap='YlOrRd',show=False)
  plt.show()

def describe_graph(graph_data):
  print(data)
  print('==============================================================')

  # Gather some statistics about the graph.
  print(f'Number of nodes: {data.num_nodes}')
  print(f'Number of edges: {data.num_edges}')
  print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
  print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
  print(f'Contains self-loops: {data.contains_self_loops()}')
  print(f'Is undirected: {data.is_undirected()}')

class DEAPDataset(InMemoryDataset):
  # 1 participant per dataset
  # Theoretically it doesn't make sense to train for all participants -> unless aiming for subject-independent classification (not atm)
  # PyG represents graphs sparsely, which refers to only holding the coordinates/values for which entries in  A  are non-zero.
  def __init__(self, root, raw_dir,processed_dir,participant_from,participant_to=None, include_edge_attr=True, undirected_graphs = True, transform=None, pre_transform=None, window_size=None):
      self._raw_dir = raw_dir
      self._processed_dir = processed_dir
      self.participant_from = participant_from
      self.participant_to = participant_from if participant_to is None else participant_to
      # Whether or not to include edge_attr in the dataset
      self.include_edge_attr = include_edge_attr
      # If true there will be 1024 links as opposed to 528
      self.undirected_graphs = undirected_graphs
      # Instantiate class to handle electrode positions
      self.electrodes = Electrodes()
      # Define the size of the windows -> 672: 12, 5.25 second windows
      self.window_size = window_size
      super(DEAPDataset, self).__init__(root, transform, pre_transform)
      self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_dir(self):
      return f'{self.root}/{self._raw_dir}'

  @property
  def processed_dir(self):
      return f'{self.root}/{self._processed_dir}'

  @property
  def raw_file_names(self):
      raw_names = [f for f in os.listdir(self.raw_dir)]
      raw_names.sort()
      return raw_names

  @property
  def processed_file_names(self):
      if not os.path.exists(self.processed_dir):
        os.makedirs(self.processed_dir)
      file_name = f'{self.participant_from}-{self.participant_to}' if self.participant_from is not self.participant_to else f'{self.participant_from}'
      return [f'deap_processed_graph.{file_name}.dataset']

  def process(self):
      # Number of nodes per graph
      n_nodes = len(self.electrodes.channel_names)

      NODE_FEATURE_N = 8064
      if NODE_FEATURE_N % self.window_size != 0:
        raise 'Error, window number of features should be divisible by window size' 

      if self.undirected_graphs:
        source_nodes, target_nodes = np.repeat(np.arange(0,n_nodes),n_nodes), np.tile(np.arange(0,n_nodes),n_nodes)
      else:
        source_nodes, target_nodes = np.tril_indices(n_nodes,n_nodes)
      edge_attr = self.electrodes.adjacency_matrix[source_nodes,target_nodes]
      
      # Remove zero weight links
      mask = np.ma.masked_not_equal(edge_attr, 0).mask
      edge_attr,source_nodes,target_nodes = edge_attr[mask], source_nodes[mask], target_nodes[mask]

      edge_attr, edge_index = torch.tensor(edge_attr), torch.tensor([source_nodes,target_nodes], dtype=torch.long)

      # List of graphs that will be written to file
      data_list = []
      pbar = tqdm(range(self.participant_from,self.participant_to+1))
      for participant_id in pbar:
        raw_name = [e for e in self.raw_file_names if str(participant_id).zfill(2) in e][0]
        pbar.set_description(raw_name)
        # Load raw file as np array
        participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
        signal_data = torch.LongTensor(participant_data['data'][:,:32,:])
        labels = torch.Tensor(participant_data['labels'])
        # Create time windows
        if self.window_size != None:
          signal_data = rearrange(signal_data,'v c (s w) -> v s c w',w = self.window_size)
        # Enumerate videos / graphs -> 
        for index_video,node_features in enumerate(signal_data):
          # Create graph
          y = torch.FloatTensor(labels[index_video]).unsqueeze(0)
          # 1 graph per window (12 per video with window size 672)
          data = Data(x=node_features,edge_attr=edge_attr,edge_index=edge_index, y=y) if self.include_edge_attr else Data(x=node_features, edge_index=edge_index, y=y)
          data_list.append(data)   
            
      data, slices = self.collate(data_list)
      torch.save((data, slices), self.processed_paths[0])