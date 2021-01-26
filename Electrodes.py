import numpy as np
import math as m
from matplotlib import pyplot as plt

'''
32 channels 
'''
class Electrodes:
  def __init__(self):
    # X, Y, Z coordinates of the electrodes
    self.positions_3d = np.array([[-27,83,-3],[-36,76,24],[-71,51,-3],[-48,59,44],
      [-33,33,74],[-78,30,27],[-87,0,-3],[-63,0,61],
      [-33,-33,74],[-78,-30,27],[-71,-51,-3],[-48,-59,44],
      [0,-63,61],[-36,-76,24],[-27,-83,-3],[0,-87,-3],
      [27,-83,-3],[36,-76,24],[48,-59,44],[71,-51,-3],
      [78,-30,27],[33,-33,74],[63,0,61],[87,0,-3],
      [78,30,27],[33,33,74],[48,59,44],[71,51,-3],
      [36,76,24],[27,83,-3],[0,63,61],[0,0,88]])
    self.channel_names = np.array(['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 
      'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 
      'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'])
    # Global connections will get a weight of -1 in the adj matrix
    self.global_connections = np.array([['Fp1','Fp2'],['AF3','AF4'],['F3','F4'],['FC5','FC6'],['T7','T8'],['CP5','CP6'],['P3','P4'],['PO3','PO4'],['O1','O2']])
    self.positions_2d = self.get_proyected_2d_positions()
    self.adjacency_matrix = self.get_adjacency_matrix()

  # Helper function for get_proyected_2d_positions
  def azim_proj(self, pos):
    [r, elev, az] = self.cart2sph(pos[0], pos[1], pos[2])
    return self.pol2cart(az, m.pi / 2 - elev)
    
  # Helper function for get_proyected_2d_positions
  def cart2sph(self, x, y, z):
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r     tant^(-1)(y/x)
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az

  # Helper function for get_proyected_2d_positions
  def pol2cart(self, theta, rho):
    return rho * m.cos(theta), rho * m.sin(theta)

  def get_proyected_2d_positions(self):
    pos_2d = np.array([self.azim_proj(pos_3d) for pos_3d in self.positions_3d])
    return pos_2d

  # Distance using projected coordinates
  # Testing needed. What is this distance when seen in 3-D? Arc of the circunference?
  def get_projected_2d_distance(self, name1, name2):
    index1, index2 = np.where(self.channel_names==name1)[0][0], np.where(self.channel_names==name2)[0][0]
    p1,p2 = self.positions_2d[index1], self.positions_2d[index2]
    incX, incY = p1[0]-p2[0] , p1[1]-p2[1]
    return m.sqrt(incX**2 + incY**2)

  def plot_2d_projection(self):
    fig, ax = plt.subplots()
    ax.scatter(self.positions_2d[:,0],self.positions_2d[:,1])
    for i,name in enumerate(self.channel_names):
        plt.text(self.positions_2d[:,0][i],self.positions_2d[:,1][i],name)
    plt.show()

  # Distance using 3d positions
  def get_3d_distance(self, name1, name2):
    index1, index2 = np.where(self.channel_names==name1)[0][0], np.where(self.channel_names==name2)[0][0]
    p1,p2 = self.positions_3d[index1], self.positions_3d[index2]
    incX, incY, incZ = p1[0]-p2[0] , p1[1]-p2[1] , p1[2]-p2[2]
    return m.sqrt(incX**2 + incY**2 + incZ**2)

  # Symetrical, contains self-loops (learnable [?])
  # Calibration constant should keep 20% of the links acording to the paper
  def get_adjacency_matrix(self, calibration_constant = 6, active_threshold = 0.1, add_global_connections = False):
    # Adjacency matrix
    # Added self-loops
    adj_mat = np.identity(len(self.channel_names))
    for i, name1 in enumerate(self.channel_names):
      for j, name2 in enumerate(self.channel_names):
        if name1 != name2:
          if add_global_connections:
            # Global connection
            global_connection = False
            for glob_con in self.global_connections:
              if (glob_con[0] ==  name1 and glob_con[1] == name2) or (glob_con[0] ==  name2 and glob_con[1] == name1):
                global_connection = True
                adj_mat[j][i] = -1
          if add_global_connections == False or not global_connection:
            # Local connection
            link_weight = (calibration_constant / self.get_3d_distance(name1,name2))
            link_weight = link_weight if link_weight > active_threshold else 0
            adj_mat[j][i] = min(1,link_weight)

    return adj_mat