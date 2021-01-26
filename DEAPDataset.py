import os
import torch
import scipy
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from tqdm import tqdm
from Electrodes import Electrodes


def plot_graph(graph_data):
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx
    from matplotlib import pyplot as plt
    graph = to_networkx(graph_data)

    plt.figure(1,figsize=(7,6)) 
    nx.draw(graph, cmap=plt.get_cmap('Set1'),node_size=75,linewidths=6)
    plt.show()

class DEAPDataset(InMemoryDataset):
  # 1 participant per dataset
  # Theoretically it doesn't make sense to train for all participants -> unless aiming for subject-independent classification (not atm)
  # PyG represents graphs sparsely, which refers to only holding the coordinates/values for which entries in  A  are non-zero.
  def __init__(self, root, raw_dir,processed_dir,participant = 0, include_edge_attr=True, undirected_graphs = True, transform=None, pre_transform=None):
      self._raw_dir = raw_dir
      self._processed_dir = processed_dir
      self.participant_from = participant
      self.participant_to = participant + 1
      # Whether or not to include edge_attr in the dataset
      self.include_edge_attr = include_edge_attr
      # If true there will be 1024 links as opposed to 528
      self.undirected_graphs = undirected_graphs
      # Instantiate class to handle electrode positions
      self.electrodes = Electrodes()
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
      return [f'deap_processed_graph.{self.participant_from}.dataset']

  def process(self):
      # Number of nodes per graph
      n_nodes = len(self.electrodes.channel_names)

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
      pbar = tqdm(self.raw_file_names)
      for i,raw_name in enumerate(pbar):
        if i in range(self.participant_from, self.participant_to):
          pbar.set_description(raw_name)
          # Load raw file as np array
          participant_data = scipy.io.loadmat(f'{self.raw_dir}/{raw_name}')
          signal_data = torch.LongTensor(participant_data['data'][:,:32,:])
          labels = torch.Tensor(participant_data['labels'])
          # Enumerate videos / graphs -> 1 graph per video
          for index_video,node_features in enumerate(signal_data):
            # Create graph
            y = torch.FloatTensor(labels[index_video]).unsqueeze(0)
            data = Data(x=node_features,edge_attr=edge_attr,edge_index=edge_index, y=y) if self.include_edge_attr else Data(x=node_features, edge_index=edge_index, y=y)
            data_list.append(data)
      data, slices = self.collate(data_list)
      torch.save((data, slices), self.processed_paths[0])