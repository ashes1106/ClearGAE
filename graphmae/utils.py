import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np

import dgl

import torch
import torch.nn as nn
from torch import optim as optim
#from tensorboardX import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import random
import scipy.sparse as sp



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    #parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--dataset", type=str, default="cora", help="cora|coaphoto|citeseer")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=True)
        
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)

    parser.add_argument('--task_type', type=str, default='nc', help='optional tesk: nc(node classification) | lp(link prediction) | gc(graph) |clu')
    parser.add_argument('--debug', type=bool, default=True, help='-')

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--tau", type=float, default=1.1)
    parser.add_argument("--differ", type=float, default=10)
    parser.add_argument("--norm_enc", type=float, default=10)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--tokenizer_type", type=str, default='gcn')
    parser.add_argument("--norm_token", type=str, default='batchnorm')
    
    args = parser.parse_args()
    return args

# ----------------------

def softmax(x):

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_similarity_neigborhood(feat,A_matrix):
    #nodes = g.ndata["feat"].cpu().numpy()
    feats  = feat.cpu().numpy()

    similarity_matrix = cosine_similarity(feats)
    #similarity_matrix = np.mean(similarity_matrix, axis=1)
    
    D = np.eye(A_matrix.shape[0])
    A_new = A_matrix - D
    similarity_neigbor = similarity_matrix * A_new

    row_sums_similarity_neigbor = np.sum(similarity_neigbor, axis=1)
    row_sums_A = np.sum(A_new, axis=1) +1e-14

    distribution_neigh_similarity = row_sums_similarity_neigbor  / row_sums_A

    return distribution_neigh_similarity

def get_similarity_difference(feat,emb_epoch,A_matrix):

    sort_compare = []
 
    feats  = feat.cpu().numpy()
    emb_epoch  = emb_epoch.cpu().numpy()
    D = np.eye(A_matrix.shape[0])
    A_new = A_matrix - D

    feat_similarity_matrix = cosine_similarity(feats) 
    emb_similarity_matrix = cosine_similarity(emb_epoch) 


    for i in range(A_new.shape[0]):
        row_real = []
        row_generator = []
        for j in range(A_new.shape[1]):
            if A_new[i][j] == 1:
                row_real.append(feat_similarity_matrix[i][j])
                row_generator.append(emb_similarity_matrix[i][j])
     
        sorted_indexes_true = np.argsort(row_real)+1
        sorted_indexes_real = np.argsort(row_generator)+1

        assert len(sorted_indexes_true)==len(sorted_indexes_real)
        correlation, _ = spearmanr(sorted_indexes_true, sorted_indexes_real)
        sort_compare.append(correlation)


    sort_compare = np.array(sort_compare)
    filtered_array = sort_compare[~np.isnan(sort_compare)]

 
    filtered_sort_compare = filtered_array.tolist()
    filtered_sort_compare_mean = np.mean(filtered_sort_compare)

    return filtered_sort_compare_mean


def get_distance(x,y):
    
        x =  x.reshape(-1,1)
        y =  y.reshape(-1,1)
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        y_scaled = scaler.fit_transform(y)

   
        dist = distance.euclidean(x_scaled.flatten() , y_scaled.flatten() )
        return dist

def plot_epoch(y_list,save_name):


    x_list = [i for i in range(1, len(y_list)+1)]

 
    plt.plot(x_list, y_list)


    plt.title("Plot Example")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    plt.savefig(save_name)
    plt.close()





def scale_feats_tensor(x):

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False)

    x -= mean
    x /= (std + 1e-7)  
    return x

def extract_indices(g):
    edge_idx_loop = g.adjacency_matrix()._indices()
    edge_idx_no_loop = dgl.remove_self_loop(g).adjacency_matrix()._indices()
    edge_idx = (edge_idx_loop, edge_idx_no_loop)
    
    return edge_idx

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph
    
    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

def extract_indices(g):
    edge_idx_loop = g.adj_tensors('coo')

    edge_idx_loop_tensors = [edge_idx_loop[1], edge_idx_loop[0]]
 
    edge_idx_loop = torch.stack(edge_idx_loop_tensors)

    edge_idx_no_loop = dgl.remove_self_loop(g).adj_tensors('coo')
    edge_idx_no_loop_tensors = [edge_idx_no_loop[1], edge_idx_no_loop[0]]

 
    edge_idx_no_loop = torch.stack(edge_idx_no_loop_tensors)
    edge_idx = (edge_idx_loop, edge_idx_no_loop)
    
    return edge_idx


# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


