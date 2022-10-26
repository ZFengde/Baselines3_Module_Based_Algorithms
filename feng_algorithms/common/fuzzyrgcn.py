import torch as th
import torch.nn as nn
import dgl
import dgl.function as fn
from feng_algorithms.common.fuzzy_logic import graph_and_etype, FuzzyInferSys, Ante_generator

# TODO, here is the place need to improve

class AnteLayer(nn.Module):
    def __init__(self):
        super(AnteLayer, self).__init__()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    def edge_func(self, edges):
        # TODO, for robot-target, we use channels method
        # but for the remains, 
        # we use the consequent weight to messure how a obstacle could affect the control
        # This can be done by introduce Mamdani fuzzy
        vector = edges.dst['h'] - edges.src['h'] 
        etypes = edges.data['rel_type'] # edge_num, batch, 1
        x1, x2 = Ante_generator(vector)  # edge_num, batch, 2
        # g.edata['rel_type']: edge_num, batch, 1, TODO, if so, could consider musk here
        # TODO, here need to see if we can change to rel based fuzzy system
        ante = FuzzyInferSys(x1, x2).to(self.device) # edge_num, batch, rules_num
        watch = th.tensor(ante).squeeze()

        return {'ante': ante}

    def forward(self, g, feat, etypes):
        g.srcdata['h'] = feat # 9, batch, input_dim 
        g.edata['rel_type'] = etypes # assigned every 
        g.apply_edges(self.edge_func)
        
        return g.edata['ante'] # 72, 6

class FuzzyRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_rules):
        super(FuzzyRGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_rules = num_rules

        # self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
        self.weight = nn.Parameter(th.Tensor(self.num_rels, self.num_rules,
                                            self.in_feat, self.out_feat))
        self.h_bias = nn.Parameter(th.Tensor(self.num_rels, self.num_rules, self.out_feat))

        # nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.h_bias)

    def message_func(self, edges):
        # TODO, here should take subgraph to do the control between robot and target
        w = self.weight[edges.data['rel_type']] # 72, 3, in * out
        h_bias = self.h_bias[edges.data['rel_type']] # 72, 3, out
        truth_values = edges.data['truth_value'] # 72, 7, 3

        # dim pre-process
        w = th.bmm(truth_values, w.view(edges.batch_size(), self.num_rules, -1)).view(-1, self.in_feat, self.out_feat) # --> 72, 7, in * out = 60

        bias = th.bmm(truth_values, h_bias)
        # TODO, here msg'd better involved both src and dst info
        msg =  th.bmm(edges.src['h'].unsqueeze(1).view(-1, 1, self.in_feat), w).view(edges.batch_size(), truth_values.shape[1], self.out_feat) # 72 * 7 * out

        return {'msg': msg + bias}

    def forward(self, g, feat, etypes, truth_value):
        with g.local_scope():
            # pass node features and etypes information
            g.ndata['h'] = feat # 9, batch, input_dim 
            g.edata['rel_type'] = etypes # assigned every 
            g.edata['truth_value'] = truth_value # 72, batch, 3

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

# node_infos = th.rand(9, 7, 6)
# g, edge_types = graph_and_etype(node_num=9)
# model = AnteLayer()
# print(model(g, node_infos, edge_types).shape)
