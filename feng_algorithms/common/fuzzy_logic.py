import numpy as np
import skfuzzy as fuzz
import dgl
import torch as th
import time

def FuzzyInferSys(x1, x2):

    x1 = x1.cpu()
    x2 = x2.cpu()
    
    x1_range = np.arange(0, 3.1, 0.1)
    x2_range = np.arange(0, 181, 1)

    x11 = fuzz.gaussmf(x1_range, 0, 0.75)
    x12 = fuzz.gaussmf(x1_range, 1.5, 0.75)
    x13 = fuzz.gaussmf(x1_range, 3, 0.75)

    x11_level = fuzz.interp_membership(x1_range, x11, x1)
    x12_level = fuzz.interp_membership(x1_range, x12, x1)
    x13_level = fuzz.interp_membership(x1_range, x13, x1)

    x21 = fuzz.gaussmf(x2_range, 0, 45)
    x22 = fuzz.gaussmf(x2_range, 90, 45)
    x23 = fuzz.gaussmf(x2_range, 180, 45)

    x21_level = fuzz.interp_membership(x2_range, x21, x2)
    x22_level = fuzz.interp_membership(x2_range, x22, x2)
    x23_level = fuzz.interp_membership(x2_range, x23, x2)

    # when the x1 or x2 out of range, but zeros is not the case
    truth_value = th.stack((th.max(th.tensor(x11_level), th.tensor(x21_level)),
                            th.max(th.tensor(x12_level), th.tensor(x22_level)),
                            th.max(th.tensor(x13_level), th.tensor(x23_level))), dim=2)
                            # here, dim=0 --> 72 * 3 * 6
                            # here, dim=1 --> 72 * 6 * 3

    return th.nn.functional.normalize(truth_value).float()

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
            elif (i==0 and 2<=j<=8):
                edge_types.append(1)

            # obstacle-robot
            elif (2<=i<=8 and j==0):
                edge_types.append(1)

            # target-obstacle
            elif (i==1 and 2<=j<=8) or (2<=i<=8 and j==1):
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

def nodes2ante(node_infos): # provide truth values based on two given nodes info
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

def Ante_generator(node1, node2):
    # 72, 7, 6
    pos_robot = node1[:, :, :2] # 72, 7
    ori = node1[:, :, 2: 4]
    vel = node1[:, :, 4: 6]
    pos_target_or_obstacle = node2[:, :, :2]
    alpha = pos_target_or_obstacle - pos_robot
    beta = vel + ori
    # here need to think about batch process
    x1 = th.linalg.norm((pos_robot - pos_target_or_obstacle), axis=2)
    x2 = angle(alpha, beta)
    return x1, x2

# v1 = th.rand(1, 2)
# v2 = th.zeros(1, 2)
# output = angle(v1, v2)
# print(output)
# obs = th.ones(22)
# if obs.dim() == 1:
#     obs = obs.unsqueeze(0)

# node_infos = obs_to_feat(obs)
# Ante_infos = nodes2ante(node_infos) # 7, 72, 2
# truth_values = FuzzyInferSys(Ante_infos)
# print(truth_values.shape)