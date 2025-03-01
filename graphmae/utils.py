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
    args = parser.parse_args()
    return args

# ----------------------

def softmax(x):

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_similarity_neigborhood(feat,A_matrix):
    #nodes = g.ndata["feat"].cpu().numpy()
    feats  = feat.cpu().numpy()
    # 计算所有节点对的余弦相似度
    similarity_matrix = cosine_similarity(feats)
    #similarity_matrix = np.mean(similarity_matrix, axis=1)
    
    D = np.eye(A_matrix.shape[0])
    A_new = A_matrix - D
    similarity_neigbor = similarity_matrix * A_new

    row_sums_similarity_neigbor = np.sum(similarity_neigbor, axis=1)
    row_sums_A = np.sum(A_new, axis=1)

    distribution_neigh_similarity = row_sums_similarity_neigbor  / row_sums_A

    return distribution_neigh_similarity

def get_similarity_difference(feat,emb_epoch,A_matrix):

    sort_compare = []
    #计算生成的嵌入表示中每一个节点与邻居节点的距离
    #计算原始图中每一个节点与邻居节点的距离
    #计算二者的KL散度
    # softmax函数
    feats  = feat.cpu().numpy()
    emb_epoch  = emb_epoch.cpu().numpy()
    D = np.eye(A_matrix.shape[0])
    A_new = A_matrix - D

    # 计算所有节点对的余弦相似度
    feat_similarity_matrix = cosine_similarity(feats) 
    emb_similarity_matrix = cosine_similarity(emb_epoch) 

    #计算排序
    # 对每一行进行操作
    for i in range(A_new.shape[0]):
        row_real = []
        row_generator = []
        for j in range(A_new.shape[1]):
            if A_new[i][j] == 1:
                row_real.append(feat_similarity_matrix[i][j])
                row_generator.append(emb_similarity_matrix[i][j])
        # 对非0元素进行排序并获取排序序号
        sorted_indexes_true = np.argsort(row_real)+1
        sorted_indexes_real = np.argsort(row_generator)+1
        # 添加到列表中
        #sorted_index_list.append(sorted_indexes + 1)  # 加1以使序号从1开始

        # 计算斯皮尔曼秩相关系数
        # 衡量两个排序的相似性
        assert len(sorted_indexes_true)==len(sorted_indexes_real)
        correlation, _ = spearmanr(sorted_indexes_true, sorted_indexes_real)
        sort_compare.append(correlation)

    # 去掉值为NaN的元素
    sort_compare = np.array(sort_compare)
    filtered_array = sort_compare[~np.isnan(sort_compare)]

    # 将过滤后的 NumPy 数组转回列表
    filtered_sort_compare = filtered_array.tolist()
    filtered_sort_compare_mean = np.mean(filtered_sort_compare)
   # print(filtered_sort_compare)  # 输出：0.7

    # 对于每个节点, 计算与其邻接节点特征的差值并应用softmax
    # n = A_new.shape[0]  # 节点数量
    # KL_all = []
    # for i in range(n):
    #     KL_line = []
    #     for j in range(n):
    #         if A_new[i][j] == 1:  # j 是 i 的邻居节点
    #             diff_1 = abs(feats[i] - feats[j])
    #             # 应用 softmax
    #             softmax_diff = softmax(diff_1)
    #             diff_2 = abs(emb_epoch[i] - emb_epoch[j])
    #             softmax_diff_2 = softmax(diff_2)
                
    #             KL_tmp = softmax_diff_2*(np.log(diff_2/diff_1))
    #             KL_line.append(KL_tmp)
    #             #print(f'节点{i}和节点{j}: 差值{diff}, Softmax差值{softmax_diff}')
    #     KL_all.append(KL_line)

    # 计算矩阵的平均值
    #KL_average = np.mean(KL_all)
    return filtered_sort_compare_mean


def get_distance(x,y):
         # 正则化
        x =  x.reshape(-1,1)
        y =  y.reshape(-1,1)
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        y_scaled = scaler.fit_transform(y)

        # 计算正则化后的距离
        dist = distance.euclidean(x_scaled.flatten() , y_scaled.flatten() )
        return dist

def plot_epoch(y_list,save_name):

    # 创建数据
    x_list = [i for i in range(1, len(y_list)+1)]

    # 绘制折线图
    plt.plot(x_list, y_list)

    # 添加标题和轴标签
    plt.title("Plot Example")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # 保存图像而不显示
    plt.savefig(save_name)
    plt.close()

def calculate(A, X):

    A = sp.coo_matrix(A)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    rowsum = np.array(A.sum(1)).clip(min=1)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    A = A.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    low = 0.5 * sp.eye(A.shape[0]) + A
    high = 0.5 * sp.eye(A.shape[0]) - A
    low = low.todense()
    high = high.todense()

    low_signal = np.dot(np.dot(low, low), X)
    high_signal = np.dot(np.dot(high, high), X)

    return low_signal, high_signal
import torch

def calculate_tensor(A, X):
    device = A.device  # If your tensors are on GPU, calculations should also be made on GPU

    # Assuming A is already a dense tensor, and its diagonal elements are zero
    A = A + torch.transpose(A, 0, 1).clone().where(A > torch.transpose(A, 0, 1), torch.tensor([-1], device=device))

    rowsum = torch.sum(A, axis=1).clamp(min=1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)

    r_mat_inv_sqrt = r_mat_inv_sqrt.float()
    A = A.float()

    A = torch.matmul(torch.matmul(A, r_mat_inv_sqrt), torch.transpose(r_mat_inv_sqrt, 0, 1))

    low = 0.5 * torch.eye(A.shape[0], device=device) + A
    high = 0.5 * torch.eye(A.shape[0], device=device) - A

    low_signal = torch.matmul(torch.matmul(low, low), X)
    high_signal = torch.matmul(torch.matmul(high, high), X)

    return low_signal, high_signal

def scale_feats_tensor(x):
    # PyTorch中的.mean()和.std()方法可以分别计算张量的平均值和标准差
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False)
    # 标准化：减去平均值，然后除以标准差
    x -= mean
    x /= (std + 1e-7)  # 加上一个小的常数防止除以0
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
    #print(edge_idx_loop[0])
    edge_idx_loop_tensors = [edge_idx_loop[1], edge_idx_loop[0]]
    # 使用 stack 函数合并张量
    edge_idx_loop = torch.stack(edge_idx_loop_tensors)
    #edge_idx_loop = edge_idx_loop.t()
    #print(edge_idx_loop)
    edge_idx_no_loop = dgl.remove_self_loop(g).adj_tensors('coo')
    edge_idx_no_loop_tensors = [edge_idx_no_loop[1], edge_idx_no_loop[0]]

    # 使用 stack 函数合并张量
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


