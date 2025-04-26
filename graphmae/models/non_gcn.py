import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair

from graphmae.utils import create_activation


class Tokenizer(nn.Module):
    def __init__(self,
               emb_dim, 
               num_layer, 
               eps, 
               JK = "last", 
               gnn_type = "gin",
               norm="batchnorm",
                 ):
        super(Tokenizer, self).__init__()
        self.num_layers = num_layer
        self.JK = JK
        self.norm = norm
      
        self.gnns = torch.nn.ModuleList()  
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(NonParaGINConv(eps))
            elif gnn_type == "gcn":
                self.gnns.append(NonParaGCNConv(eps))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(NonParaGraphSAGEConv(eps, aggr='mean'))
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(self.norm(emb_dim))

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
                h = self.gnns[l](g, h)
                h = self.batch_norms[l](h)
                hidden_list.append(h)

                ### Different implementations of Jk-concat
        if self.JK == "concat":
            g_tokens = torch.cat(hidden_list, dim = 1)
        elif self.JK == 'first_cat':
            g_tokens = torch.cat([hidden_list[0], hidden_list[-1]], dim = 1)
        elif self.JK == "last":
            g_tokens = hidden_list[-1]
        elif self.JK == "max":
            hidden_list = [h.unsqueeze_(0) for h in hidden_list]
            g_tokens = torch.max(torch.cat(hidden_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            hidden_list = [h.unsqueeze_(0) for h in hidden_list]
            g_tokens = torch.sum(torch.cat(hidden_list, dim = 0), dim = 0)[0]


        return g_tokens





    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class NonParaGCNConv(nn.Module):
    def __init__(self, eps):
        super(NonParaGCNConv, self).__init__()
        self.eps = eps

    def forward(self, graph, x):
        with graph.local_scope():
            # 添加自环边
            #graph = dgl.add_self_loop(graph)
            
            # 计算度矩阵的平方根的倒数
            degs = graph.out_degrees().float().clamp(min=1)  # 防止除零
            norm = torch.pow(degs, -0.5)
            norm = norm.to(x.device).unsqueeze(1)  # (num_nodes, 1)
            
            # 将归一化系数存储到节点和边上
            graph.ndata['norm'] = norm
            graph.apply_edges(fn.u_mul_v('norm', 'norm', 'edge_norm'))  # edge_norm = norm[u] * norm[v]
            
            # 消息传递
            graph.ndata['x'] = x
            graph.update_all(
                fn.u_mul_e('x', 'edge_norm', 'm'),  # 消息函数：x_j * edge_norm
                fn.sum('m', 'h')                   # 聚合函数：求和
            )
            
            # 获取聚合结果
            h = graph.ndata['h']
            
            # 残差连接
            return h + x * self.eps


class NonParaGINConv(nn.Module):
    def __init__(self, eps=0, aggregator_type="sum"):
        super().__init__()
        self.eps = eps  # 控制自环特征权重的标量
        self._aggregator_type = aggregator_type

        # 设置聚合函数
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))

    def forward(self, graph, feat):
        with graph.local_scope():
            # 定义消息传递函数
            aggregate_fn = fn.copy_u('h', 'm')

            # 将特征扩展为 (feat_src, feat_dst) 对
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # 消息传递和聚合
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']

            return rst