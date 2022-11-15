import torch as th
import torch.nn as nn
import dgl
import dgl.function as fn
from feng_algorithms.common.fuzzy_logic import Ante_generator, graph_and_etype
import numpy as np
import time

class gaussmf():
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def ante(self, x):
        return th.exp(-((x - self.mean)**2.) / (2 * self.sigma **2.))

class TSFuzzyLayer(nn.Module): # -> coupling_degree, truth_value
    def __init__(self):
        super(TSFuzzyLayer, self).__init__()
        self.rules_num = 9
        # 2 * 9
        self.sub_systems_mat = nn.Parameter(th.tensor([[0, -0.05, -0.2, -0.1, -0.25, -0.02, -0.2, -0.05, 0], 
                                                        [0, -0.0011, -0.0022, -0.0022, 0, -0.00044, -0.0011, -0.0011, 0]]), requires_grad=False)
        self.sub_systems_bias = nn.Parameter(th.tensor([1, 1, 0.7, 1, 1, 0.22, 0.9, 0.35, 0]), requires_grad=False)
        self._init_rules()

    def edge_func(self, edges):

        # preprocessing
        vector = edges.dst['h'] - edges.src['h'] # edge_num, batch, input_dim
        x1, x2 = Ante_generator(vector)  # edge_num, batch, 2
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        # 3 * 3 --> 9
        truth_value = self.ante_process(x1, x2) # 72, 7, 9, as coeffient

        # stack x1, x2 together
        premises = th.stack((x1, x2), dim=2).view(-1, 2).float() # 9, 72*7, 2
        # n, 2 * 2, 9 --> n, 9, use consequent matrix generate consequence
        consequence = th.matmul(premises, self.sub_systems_mat) + self.sub_systems_bias
        # which is the output of different consequent matrix, but vectorized
        consequence = consequence.view(x1.shape[0], x1.shape[1], self.rules_num)
        # normalized and output 
        coupling_degree = th.sum((truth_value * consequence), dim=2) / th.sum(truth_value, dim=2)

        # 1 / 9
        return {'coupling_degree': coupling_degree, 'truth_value': truth_value}

    def forward(self, g, feat):
        g.srcdata['h'] = feat # 9, batch, input_dim 
        g.apply_edges(self.edge_func)
        
        return g.edata['coupling_degree'], g.edata['truth_value'] # 72, batch

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

class FuzzyRGCNLayer(nn.Module): # using antecedants to update node features
    def __init__(self, in_feat, out_feat, num_rels, num_rules):
        super(FuzzyRGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_rules = num_rules

        # self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
        self.weight = nn.Parameter(th.Tensor(self.num_rels, self.in_feat, self.out_feat))
        self.h_bias = nn.Parameter(th.Tensor(self.num_rels, self.out_feat))

        # 9, 6, 10
        self.weight_robot_target = nn.Parameter(th.Tensor(self.num_rules, self.in_feat, self.out_feat))
        # nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.weight_robot_target, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.h_bias)

    def message_func(self, edges):
        # assign parameters according to the relation type
        w = self.weight[edges.data['rel_type']].view(edges.batch_size(), 1, -1) # 72, in * out
        h_bias = self.h_bias[edges.data['rel_type']].unsqueeze(1) # edge_num, out_feat

        coupling_degrees = edges.data['coupling_degree'].unsqueeze(2) # edge_num, batch, 1
        # but operation cause can't compute gradient
        # TODO, should be conducted relational softmax here
        coupling_degrees[self.ID] = 1.
        # this is only for robot-target 
        ante = edges.data['truth_value'].view(-1, self.num_rules) # edge_num, batch, rule_num --> 72* 7, 9 

        # multiple weight with a coupling degree and mulitiple robot-target weight with ante
        weighted_w = th.bmm(coupling_degrees, w).view(edges.batch_size(), -1, self.in_feat, self.out_feat) # edge_num * batch, in, out
        # edge_num, batch, in_feat, out_feat, in this operation:
        # edge_num, batch, num_rules * num_rules, in_feat, out_feat, i.e., weigted sum of different weights
        # --> edge_num, batch, in_feat, out_feat
        ante_w = th.matmul(ante, self.weight_robot_target.view(self.num_rules, -1)).view(edges.batch_size(), -1, self.in_feat, self.out_feat)

        # replace robot-target weight with ante_w, since we treat them differently
        weighted_w[self.ID] = ante_w[self.ID] # edge_num, batch, in_feat, out_feat

        msg =  th.bmm(edges.src['h'].view(-1, 1, self.in_feat), weighted_w.view(-1, self.in_feat, self.out_feat)).view(edges.batch_size(), -1, self.out_feat)

        # TODO, but we only have one bias system
        msg += th.matmul(coupling_degrees, h_bias) # edge_num, batch, out_feat
        
        return {'msg': msg} # edge_num, batch, out_feat

    def forward(self, g, feat, etypes, coupling_degree, truth_value, edge_sg_ID):
        with g.local_scope(): 
            # pass node features and etypes information
            g.ndata['h'] = feat # 9, batch, input_dim 
            g.edata['rel_type'] = etypes # assigned every 
            g.edata['coupling_degree'] = coupling_degree # edge_num, batch, 1
            g.edata['truth_value'] = truth_value # edge_num, batch, 9
            
            # TODO, here is generate edge type depedent-subgraphs
            edge_sg_ID
            sg = dgl.node_subgraph(g, [0, 1])

            self.ID = sg.edata[dgl.EID]
            # message passing
            g.update_all(self.message_func, fn.sum('msg', 'h'))

            return g.dstdata['h']

class FuzzyRGCN(nn.Module):
    def __init__(self, input_dim, h_dim, out_dim, num_rels, num_rules):
        super(FuzzyRGCN, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_rules = num_rules

        self.ante_layer = TSFuzzyLayer()
        self.layer1 = FuzzyRGCNLayer(self.input_dim, self.h_dim, self.num_rels, self.num_rules)
        self.layer2 = FuzzyRGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_rules)

    def forward(self, g, feat, etypes, edge_sg_ID):
        
        coupling_degree, truth_value = self.ante_layer(g, feat) # here spend too many time
        x = th.tanh(self.layer1(g, feat, etypes, coupling_degree, truth_value, edge_sg_ID))
        x = self.layer2(g, x, etypes, coupling_degree, truth_value, edge_sg_ID) # node_num, batch, out_dim
        
        return x

# device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# node_infos = th.rand(9, 7, 6).to(device)
# g, edge_types = graph_and_etype(node_num=9)

# g = g.to(device)
# edge_types = edge_types.to(device)

# model = FuzzyRGCN(6, 10, 8, 4, 9).to(device)

# print(model(g, node_infos, edge_types).shape)
