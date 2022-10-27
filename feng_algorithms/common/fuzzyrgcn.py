import torch as th
import torch.nn as nn
import dgl
import dgl.function as fn
from feng_algorithms.common.fuzzy_logic import FuzzyInferSys, Ante_generator, graph_and_etype
from skfuzzy import control as ctrl
import numpy as np
import skfuzzy as fuzz

# TODO, here is the place need to improve

class AnteLayer(nn.Module):
    def __init__(self):
        super(AnteLayer, self).__init__()
        self._init_fuzzy_system()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    def edge_func(self, edges):

        '''
        TODO,
        1. for robot-target, we use channels method
        
        2. but for the remains, 
        we use the consequent weight to messure how a obstacle could affect the control
        This can be done by introduce Mamdani fuzzy

        '''
        # TODO, what we want

        vector = edges.dst['h'] - edges.src['h'] 
        etypes = edges.data['rel_type'] # edge_num, batch, 1
        x1, x2 = Ante_generator(vector)  # edge_num, batch, 2
        # g.edata['rel_type']: edge_num, batch, 1, TODO, if so, could consider musk here
        ante = FuzzyInferSys(x1, x2) # edge_num, batch, rules_num
        self.coupling_degree_system.input['x1'] = np.array(x1.cpu()) # 72, 7
        self.coupling_degree_system.input['x2'] = np.array(x2.cpu()) # 72, 7
        self.coupling_degree_system.compute()
        coupling_degree = th.tensor(self.coupling_degree_system.output['coupling_degree']).float().to(self.device)

        return {'coupling_degree': coupling_degree}

    def forward(self, g, feat, etypes):
        g.srcdata['h'] = feat # 9, batch, input_dim 
        g.edata['rel_type'] = etypes # assigned every 
        g.apply_edges(self.edge_func)
        
        return g.edata['coupling_degree'] # 72, batch

    def _init_fuzzy_system(self):
        x1 = ctrl.Antecedent(np.arange(0, 4.1, 0.1), 'x1')
        x2 = ctrl.Antecedent(np.arange(0, 181, 1), 'x2')
        coupling_degree = ctrl.Consequent(np.arange(-0.3, 1.31, 0.01), 'coupling_degree')

        x1['XS'] = fuzz.gaussmf(x1.universe, 0, 1)
        x1['S'] = fuzz.gaussmf(x1.universe, 1, 1)
        x1['M'] = fuzz.gaussmf(x1.universe, 2, 1)
        x1['L'] = fuzz.gaussmf(x1.universe, 3, 1)
        x1['XL'] = fuzz.gaussmf(x1.universe, 4, 1)

        x2['XS'] = fuzz.gaussmf(x2.universe, 0, 45)
        x2['S'] = fuzz.gaussmf(x2.universe, 45, 45)
        x2['M'] = fuzz.gaussmf(x2.universe, 90, 45)
        x2['L'] = fuzz.gaussmf(x2.universe, 135, 45)
        x2['XL'] = fuzz.gaussmf(x2.universe, 180, 45)

        coupling_degree['XS'] = fuzz.gaussmf(coupling_degree.universe, -0.3, 0.3)
        coupling_degree['S'] = fuzz.gaussmf(coupling_degree.universe, 0.1, 0.3)
        coupling_degree['M'] = fuzz.gaussmf(coupling_degree.universe, 0.5, 0.3)
        coupling_degree['L'] = fuzz.gaussmf(coupling_degree.universe, 0.9, 0.3)
        coupling_degree['XL'] = fuzz.gaussmf(coupling_degree.universe, 1.3, 0.3)

        rule1 = ctrl.Rule(antecedent=((x1['M'] & x2['XL'])|
                                    (x1['L'] & x2['XL'])|
                                    (x1['L'] & x2['L'])|
                                    (x1['XL'] & x2['L'])|
                                    (x1['XL'] & x2['L'])|
                                    (x1['XL'] & x2['XL'])),
                        consequent=coupling_degree['XS'])

        rule2 = ctrl.Rule(antecedent=((x1['S'] & x2['XL'])|
                                    (x1['M'] & x2['L'])|
                                    (x1['L'] & x2['M'])|
                                    (x1['XL'] & x2['S'])),
                        consequent=coupling_degree['S'])

        rule3 = ctrl.Rule(antecedent=(
                                    (x1['XS'] & x2['XL'])|
                                    (x1['S'] & x2['L'])|
                                    (x1['M'] & x2['M'])|
                                    (x1['L'] & x2['S'])|
                                    (x1['XL'] & x2['XS'])),
                        consequent=coupling_degree['M'])

        rule4 = ctrl.Rule(antecedent=(
                                    (x1['XS'] & x2['L'])|
                                    (x1['S'] & x2['M'])|
                                    (x1['M'] & x2['S'])|
                                    (x1['L'] & x2['XS'])),
                        consequent=coupling_degree['L'])

        rule5 = ctrl.Rule(antecedent=(
                                    (x1['XS'] & x2['M'])|
                                    (x1['XS'] & x2['S'])|
                                    (x1['XS'] & x2['XS'])|
                                    (x1['S'] & x2['S'])|
                                    (x1['S'] & x2['XS'])|
                                    (x1['M'] & x2['XS'])),
                        consequent=coupling_degree['XL'])

        coupling_degree_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.coupling_degree_system = ctrl.ControlSystemSimulation(coupling_degree_ctrl)
        

class FuzzyRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_rules):
        super(FuzzyRGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_rules = num_rules

        # self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
        self.weight = nn.Parameter(th.Tensor(self.num_rels, self.in_feat, self.out_feat))
        self.weight_robot_target = nn.Parameter(th.Tensor(self.num_rules, self.in_feat, self.out_feat))
        self.h_bias = nn.Parameter(th.Tensor(self.num_rels, self.num_rules, self.out_feat))

        # nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.h_bias)

    def message_func(self, edges):
        # TODO, here should take subgraph to do the control between robot and target
        w = self.weight[edges.data['rel_type']].view(edges.batch_size(), 1, -1) # 72, in * out
        coupling_degrees = edges.data['coupling_degree'].unsqueeze(2) # 72, 7
        h_bias = self.h_bias[edges.data['rel_type']] # 72, 3, out

        # TODO, here is the place of next step
        weighted_w = th.bmm(coupling_degrees, w).view(-1, self.in_feat, self.out_feat) # edge_num * batch, in, out


        # bias = th.bmm(coupling_degrees, h_bias)
        # TODO, here msg'd better involved both src and dst info
        msg =  th.bmm(edges.src['h'].view(-1, 1, self.in_feat), weighted_w).view(edges.batch_size(), -1, self.out_feat) # 72 * 7 * out

        return {'msg': msg}

    def forward(self, g, feat, etypes, coupling_degree):
        with g.local_scope():
            # pass node features and etypes information
            g.ndata['h'] = feat # 9, batch, input_dim 
            g.edata['rel_type'] = etypes # assigned every 
            g.edata['coupling_degree'] = coupling_degree # 72, batch, 3

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

        self.ante_layer = AnteLayer()
        self.layer1 = FuzzyRGCNLayer(self.input_dim, self.h_dim, self.num_rels, self.num_rules)
        self.layer2 = FuzzyRGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_rules)

    def forward(self, g, feat, etypes):
        truth_value = self.ante_layer(g, feat, etypes)
        x = th.tanh(self.layer1(g, feat, etypes, truth_value))
        x = self.layer2(g, x, etypes, truth_value)
        return x

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
node_infos = th.rand(9, 7, 6).to(device)
g, edge_types = graph_and_etype(node_num=9)

g = g.to(device)
edge_types = edge_types.to(device)

model = FuzzyRGCN(6, 10, 8, 4, 9).to(device)

print(model(g, node_infos, edge_types).shape)
