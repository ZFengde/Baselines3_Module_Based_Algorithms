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
    
class TS_Fuzzy(nn.Module):
    def __init__(self, rules_num):
        super().__init__()

        self.rules_num = rules_num
        self.sub_systems_mat = th.tensor([[0, -0.05, -0.2, -0.1, -0.25, -0.02, -0.2, -0.05, 0], 
                                        [0, -0.0011, -0.0022, -0.0022, 0, -0.00044, -0.0011, -0.0011, 0]])
        self.sub_systems_bias = th.tensor([1, 1.05, 0.7, 1, 1, 0.22, 0.9, 0.35, 0])
        self._init_rules()

    def forward(self, x1, x2): # x1 = 72, 7
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        truth_value = self.ante_process(x1, x2) # 72, 7, 9, as coeffient

        premises = th.stack((x1, x2), dim=2).view(-1, 2).float() # 9, 72*7, 2
        consequence = th.matmul(premises, self.sub_systems_mat) + self.sub_systems_bias
        consequence = consequence.view(x1.shape[0], x1.shape[1], self.rules_num)

        output = th.sum((truth_value * consequence), dim=2) / th.sum(truth_value, dim=2)
        return truth_value, output
    def ante_process(self, x1, x2) : #-> truth_values
        # see if here can be batch operations, but not very important
        x1_s_level = self.x1_s.ante(x1)
        x1_m_level = self.x1_m.ante(x1)
        x1_l_level = self.x1_l.ante(x1)

        x2_s_level = self.x2_s.ante(x2)
        x2_m_level = self.x2_m.ante(x2)
        x2_l_level = self.x2_l.ante(x2)

        truth_values = th.stack((th.min(x1_s_level, x2_s_level),
                         th.min(x1_s_level, x2_m_level),
                         th.min(x1_s_level, x2_l_level), 
                         th.min(x1_m_level, x2_s_level), 
                         th.min(x1_m_level, x2_m_level), 
                         th.min(x1_m_level, x2_l_level),
                         th.min(x1_l_level, x2_s_level),
                         th.min(x1_l_level, x2_m_level),
                         th.min(x1_l_level, x2_l_level)), dim=2) 

        return truth_values

    def _init_rules(self):
        self.x1_s = gaussmf(0, 0.75) # mean and sigma
        self.x1_m = gaussmf(2, 0.75)
        self.x1_l = gaussmf(4, 0.75)

        self.x2_s = gaussmf(0, 30) # mean and sigma
        self.x2_m = gaussmf(90, 30) # mean and sigma
        self.x2_l = gaussmf(180, 30) # mean and sigma

def graph_and_etype(node_num): # -> graph, edge_types
    edge_src = []
    edge_dst = []
    edge_types = []
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

            # robot-obstacle
            elif (i==0 and 2<=j) or (2<=i and j==0):
                edge_types.append(1)

            # target-obstacle
            elif (i==1 and 2<=j) or (2<=i and j==1):
                edge_types.append(2)

            # obstacle-obstacle
            else:
                edge_types.append(3)

            # see if here could be changed to batch operation
    return dgl.graph((edge_src, edge_dst)), th.tensor(edge_types)

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
