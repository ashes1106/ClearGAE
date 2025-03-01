
from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset,
    CoauthorPhysicsDataset, 
    CoauthorCSDataset,
    AmazonCoBuyPhotoDataset,
    AmazonCoBuyComputerDataset
)
#from ogb.nodeproppred import DglNodePropPredDataset
#from dgl.data.ppi import PPIDataset
#from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler



GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
   # "ogbn-arxiv": DglNodePropPredDataset,
   # "ogbn-papers100M": DglNodePropPredDataset,
   # "ogbn-products": DglNodePropPredDataset,
    "coacs": CoauthorCSDataset,
    "coaphysics": CoauthorPhysicsDataset,
    "coaphoto":AmazonCoBuyPhotoDataset,
    "coacomputer":AmazonCoBuyComputerDataset
}


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats




def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats



def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    mask = mask.bool()
    return mask


# def load_dataset(dataset_name):
#     assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
#     if dataset_name.startswith("ogbn"):
#         dataset = GRAPH_DICT[dataset_name](dataset_name)
#     else:
#         dataset = GRAPH_DICT[dataset_name]()

#     if dataset_name == "ogbn-arxiv":
#         graph, labels = dataset[0]
#         num_nodes = graph.num_nodes()

#         split_idx = dataset.get_idx_split()
#         train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
#         graph = preprocess(graph)

#         if not torch.is_tensor(train_idx):
#             train_idx = torch.as_tensor(train_idx)
#             val_idx = torch.as_tensor(val_idx)
#             test_idx = torch.as_tensor(test_idx)

#         feat = graph.ndata["feat"]
#         feat = scale_feats(feat)
#         graph.ndata["feat"] = feat

#         train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
#         val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
#         test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
#         graph.ndata["label"] = labels.view(-1)
#         graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
#     else:
#         graph = dataset[0]
#         graph = graph.remove_self_loop()
#         graph = graph.add_self_loop()
#         if dataset_name.startswith("coa"):            
#             size = graph.ndata['feat'].shape[0]  
#             idxs = [i for i in range(size)]            
#             train_ratio, val_ratio = 0.1, 0.2
#             split_train = int(size * train_ratio)            
#             split_val = int(size * val_ratio)
            
#             train_idx = idxs[:split_train]                   
#             mask = torch.zeros(size, dtype=bool)
#             mask[train_idx] = True
#             graph.ndata['train_mask'] = mask

#             val_idx = idxs[split_train:split_val]
#             mask = torch.zeros(size, dtype=bool)
#             mask[val_idx] = True
#             graph.ndata['val_mask'] = mask

#             test_idx = idxs[split_val:]
#             mask = torch.zeros(size, dtype=bool)
#             mask[test_idx] = True
#             graph.ndata['test_mask'] = mask

#     num_features = graph.ndata["feat"].shape[1]
#     num_classes = dataset.num_classes
#     return graph, (num_features, num_classes)


def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()
        
    if dataset_name.startswith("ogbn"):
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:        
        graph = dataset[0]
        if dataset_name.startswith("coa"):            
            print("------dataset------", dataset_name)
            size = graph.ndata['feat'].shape[0]  
            labels = graph.ndata['label']
            indices = torch.arange(len(labels))

            train_ratio = 0.1
            val_ratio = 0.1
            test_ratio = 0.8

            N = graph.number_of_nodes()
            train_num = int(N * train_ratio)
            val_num = int(N * (train_ratio + val_ratio))

            idx = np.arange(N)
            # idx = torch.arange(N)
            np.random.shuffle(idx)

            train_idx = idx[:train_num]
            val_idx = idx[train_num:val_num]
            test_idx = idx[val_num:]

            train_mask = sample_mask(train_idx, N)
            val_mask = sample_mask(val_idx, N)
            test_mask = sample_mask(test_idx, N)

            graph.ndata['train_mask'] = train_mask
            graph.ndata['val_mask'] = val_mask
            graph.ndata['test_mask'] = test_mask

        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)
