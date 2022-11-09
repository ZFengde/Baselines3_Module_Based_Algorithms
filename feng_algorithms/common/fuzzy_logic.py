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

    def ante_process(self, x1, x2):
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

def FuzzyInferSys(x1, x2):
    
    '''
    relationships: 
    0: robot-target, 1: robot-obstacle
    2: target-obstacle, 3:obstacle-obstacle
    '''

    x1 = th.clip(x1, 0, 3)
    x2 = th.clip(x2, 0, 180)
    
    x1_range = np.arange(0, 3.1, 0.1)
    x2_range = np.arange(0, 181, 1)
    coupling_range = np.arange(0, 1, 0.01)

    x1_s = fuzz.gaussmf(x1_range, 0, 0.75)
    x1_m = fuzz.gaussmf(x1_range, 1.5, 0.75)
    x1_l = fuzz.gaussmf(x1_range, 3, 0.75)

    x1_s_level = fuzz.interp_membership(x1_range, x1_s, x1)
    x1_m_level = fuzz.interp_membership(x1_range, x1_m, x1)
    x1_l_level = fuzz.interp_membership(x1_range, x1_l, x1)

    x2_s = fuzz.gaussmf(x2_range, 0, 45)
    x2_m = fuzz.gaussmf(x2_range, 90, 45)
    x2_l = fuzz.gaussmf(x2_range, 180, 45)

    x2_s_level = fuzz.interp_membership(x2_range, x2_s, x2)
    x2_m_level = fuzz.interp_membership(x2_range, x2_m, x2)
    x2_l_level = fuzz.interp_membership(x2_range, x2_l, x2)
    
    coupling_s = fuzz.gaussmf(coupling_range, 0, 0.2)
    coupling_m = fuzz.gaussmf(coupling_range, 0.5, 0.2)
    coupling_l = fuzz.gaussmf(coupling_range, 1, 0.2)

    a = th.max(th.tensor(x1_s_level), th.tensor(x2_s_level))
    # edge_num, batch, rules_num
    active_rules = th.cat((th.max(th.tensor(x1_s_level), th.tensor(x2_s_level)), # l
                            th.max(th.tensor(x1_s_level), th.tensor(x2_m_level)), # l
                            th.max(th.tensor(x1_s_level), th.tensor(x2_l_level)), # m
                            th.max(th.tensor(x1_m_level), th.tensor(x2_s_level)), # l
                            th.max(th.tensor(x1_m_level), th.tensor(x2_m_level)), # m
                            th.max(th.tensor(x1_m_level), th.tensor(x2_l_level)), # l
                            th.max(th.tensor(x1_l_level), th.tensor(x2_s_level)), # s
                            th.max(th.tensor(x1_l_level), th.tensor(x2_m_level)), # s
                            th.max(th.tensor(x1_l_level), th.tensor(x2_l_level))), dim=2) # s

    a = th.max(active_rules[:, :][0, 3, 6])
    coupling_s_level = th.min(coupling_s, th.max(active_rules[0, 3, 6]).squeeze()) # coupling_s edge, batch, 1
    coupling_m_level = th.min(coupling_m, th.max(active_rules[1, 4, 7]))
    coupling_l_level = th.min(coupling_l, th.max(active_rules[2, 5, 8]))
    # return th.nn.functional.normalize((truth_value), dim=2).float()
    return 
    
def graph_and_etype(node_num): # generate graph, etypes
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

def obs_to_feat(obs): # transfer observation into node features form
    # obs_size = 6 + 2 + 7*2 = 22
    
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    if obs.dim() == 2:
        m = th.nn.ZeroPad2d((0, 4, 0, 0))
        robot_info = obs[:, :6]
        target_info = m(obs[:, 6: 8])
        obstacle_infos = m(obs[:, 8:].view(-1, 7, 2))

        node_infos = th.cat((robot_info.unsqueeze(1), target_info.unsqueeze(1), obstacle_infos), dim=1)
        return node_infos

def nodes2ante(node_infos): # Discard method
    # node1 always refer to moving object or satatic object while node2 refer to static object
    Ante = []

    for i in range(node_infos.shape[1]):
        for j in range(node_infos.shape[1]):
            if i == j:
                continue

            # robot-target
            if (i==0 and j==1) or (i==1 and j==0):
                x1, x2 = Ante_generator(node_infos[:, 0, :], node_infos[:, 1, :], angle_include=True)
                Ante.append(th.stack((x1, x2), dim=1)) # 4, 2

            # robot-obstacle
            elif (i==0 and 2<=j<=8):
                x1, x2 = Ante_generator(node_infos[:, 0, :], node_infos[:, j, :], angle_include=True)
                Ante.append(th.stack((x1, x2), dim=1))

            # obstacle-robot
            elif (2<=i<=8 and j==0):
                x1, x2 = Ante_generator(node_infos[:, 0, :], node_infos[:, i, :], angle_include=True)
                Ante.append(th.stack((x1, x2), dim=1))

            # target-obstacle and obstacle-obstacle
            else:
                x1, x2 = Ante_generator(node_infos[:, i, :], node_infos[:, j, :], angle_include=False)
                Ante.append(th.stack((x1, x2), dim=1))

    return th.stack(Ante, dim=0)

def angle(v1, v2): # calculate angle between two give vectors
    # 72, 7, 2, 72, 7, 2
    epsilon =1e-8
    cos = th.nn.CosineSimilarity(dim=2, eps=1e-6)
    cos_value = cos(v1, v2)
    # This is for safe acos, reference to note
    radian = th.acos(th.clamp(cos_value, -1 + epsilon, 1 - epsilon))
    degree = th.rad2deg(radian)
    return degree

def Ante_generator(vector):

    alpha = vector[:, :, :2]
    beta = vector[:, :, 2: 4] + vector[:, :, 4: 6]
    x1 = th.linalg.norm((alpha), axis=2)
    x2 = angle(-alpha, beta)
    return x1, x2

# x1 = th.rand(1, 1, 1)
# x2 = th.rand(1, 1, 1)
# truth_values = FuzzyInferSys(x1, x2)
# print(truth_values.shape)