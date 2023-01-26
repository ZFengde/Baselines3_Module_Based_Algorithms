import numpy as np
import skfuzzy as fuzz
import dgl
import torch as th
import time
import torch.nn as nn

class gaussmf():
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def ante(self, x):
        return th.exp(-((x - self.mean)**2.) / (2 * self.sigma **2.))

def graph_and_etype(node_num): # -> graph, edge_types
    edge_src = []
    edge_dst = []
    edge_types = []
    ID_indicator = 0
    edge_type_ID = [[], [], [], []]

    for i in range(node_num):
        for j in range(node_num):

            if i == j:
                continue
            edge_src.append(i)
            edge_dst.append(j)
            '''
            relationships: 
            0: robot-target, 1: robot-obstacle
            2: target-obstacle, 3:obstacle-obstacle
            '''
            # robot-target
            if (i==0 and j==1) or (i==1 and j==0):
                edge_types.append(0)
                edge_type_ID[0].append(ID_indicator)

            # robot-obstacle
            elif (i==0 and 2<=j) or (2<=i and j==0):
                edge_types.append(1)
                edge_type_ID[1].append(ID_indicator)

            # target-obstacle
            elif (i==1 and 2<=j) or (2<=i and j==1):
                edge_types.append(2)
                edge_type_ID[2].append(ID_indicator)

            # obstacle-obstacle
            else:
                edge_types.append(3)
                edge_type_ID[3].append(ID_indicator)
            
            ID_indicator += 1

            # see if here could be changed to batch operation
    return dgl.graph((edge_src, edge_dst)), th.tensor(edge_types), edge_type_ID

def obs_to_feat(obs): # -> node_infos
    # obs_size = 6 + 2 + 3*2 = 22 14
    
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    if obs.dim() == 2:
        m = th.nn.ZeroPad2d((0, 4, 0, 0))
        obs_num = int((obs.shape[1] - 8) / 2)
        robot_info = obs[:, :6]
        target_info = m(obs[:, 6: 8]) # 6, 6
        obstacle_infos = m(obs[:, 8:].view(-1, obs_num, 2))

        node_infos = th.cat((robot_info.unsqueeze(1), target_info.unsqueeze(1), obstacle_infos), dim=1)
        return node_infos

def angle(v1, v2): # -> degree
    # 72, 7, 2, 72, 7, 2
    epsilon =1e-8
    cos = th.nn.CosineSimilarity(dim=2, eps=1e-6)
    cos_value = cos(v1, v2)
    # This is for safe acos, reference to note
    radian = th.acos(th.clamp(cos_value, -1 + epsilon, 1 - epsilon))
    degree = th.rad2deg(radian)
    return degree

def Ante_generator(vector): # -> x1, x2

    alpha = vector[:, :, :2]
    beta = vector[:, :, 2: 4] + vector[:, :, 4: 6]
    x1 = th.linalg.norm((alpha), axis=2)
    x2 = angle(-alpha, beta)
    return x1, x2
