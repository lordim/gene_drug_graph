import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, PrefetchLoader

from .sample_negatives import SampleNegatives

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_counts(data: HeteroData) -> tuple[int, int, dict]:
    num_sources = len(data["source"].node_id)
    num_targets = len(data["target"].node_id)
    num_features = data.num_node_features
    return num_sources, num_targets, num_features


def get_all_st_edges(data: HeteroData, pretrain_source: bool) -> np.ndarray:
    if pretrain_source:
        msgs = data["source", "similar", "source"]["edge_index"]
        sups = data["source", "similar", "source"]["edge_label_index"]
    
    else:
        msgs = data["binds"]["edge_index"]
        sups = data["binds"]["edge_label_index"]
    edges = torch.concat([msgs, sups], axis=1).cpu().numpy()
    return edges


def load_node_features(source_path: str, target_path: str) -> HeteroData:
    source = pd.read_parquet(source_path)
    target = pd.read_parquet(target_path)

    data = HeteroData()
    data["source"].node_id = torch.arange(len(source))
    data["source"].x = torch.from_numpy(source.values).to(torch.float32)
    data["target"].node_id = torch.arange(len(target))
    data["target"].x = torch.from_numpy(target.values).to(torch.float32)
    return data


def load_bipartite_edges(
    labels_path: str, message: list[str], supervision: list[str]
) -> tuple[torch.tensor, torch.tensor]:
    """Load messages and supervision edges in torch tensors"""
    edges = pd.read_parquet(labels_path)
    msgs = edges.query("subset.isin(@message)")
    msgs = torch.tensor(msgs[["source", "target"]].values.T, dtype=torch.long)
    sups = edges.query("subset.isin(@supervision)")
    sups = torch.tensor(sups[["source", "target"]].values.T, dtype=torch.long)
    return msgs, sups


def load_s_expanded_edges(
    s_s_labels_path: str,
    s_t_labels_path: str,
    message: list[str],
    supervision: list[str],
) -> tuple[torch.tensor, torch.tensor]:
    """Load messages and supervision edges in torch tensors"""
    s_s_edges = pd.read_parquet(s_s_labels_path)
    s_t_edges = pd.read_parquet(s_t_labels_path)

    # set source target edges as message and supervision training edges
    msgs = s_t_edges.query("subset.isin(@message)")
    msgs = torch.tensor(msgs[["source", "target"]].values.T, dtype=torch.long)
    sups = s_t_edges.query("subset.isin(@supervision)")
    sups = torch.tensor(sups[["source", "target"]].values.T, dtype=torch.long)

    # only include message source/source and target_target edges
    if message == ["message"]:
        s_s_msgs = s_s_edges.query("subset.isin(@supervision)")

    else:
        s_s_msgs = s_s_edges.query("subset.isin(@message)")

    s_s_msgs = torch.tensor(
        s_s_msgs[["source_a", "source_b"]].values.T, dtype=torch.long
    )

    return msgs, sups, s_s_msgs


def load_t_expanded_edges(
    s_t_labels_path: str,
    t_t_labels_path: str,
    message: list[str],
    supervision: list[str],
) -> tuple[torch.tensor, torch.tensor]:
    """Load messages and supervision edges in torch tensors"""
    s_t_edges = pd.read_parquet(s_t_labels_path)
    t_t_edges = pd.read_parquet(t_t_labels_path)

    # set source target edges as message and supervision training edges
    msgs = s_t_edges.query("subset.isin(@message)")
    msgs = torch.tensor(msgs[["source", "target"]].values.T, dtype=torch.long)
    sups = s_t_edges.query("subset.isin(@supervision)")
    sups = torch.tensor(sups[["source", "target"]].values.T, dtype=torch.long)

    # only include message source/source and target_target edges
    if message == ["message"]:
        t_t_msgs = t_t_edges.query("subset.isin(@supervision)")

    else:
        t_t_msgs = t_t_edges.query("subset.isin(@message)")

    t_t_msgs = torch.tensor(
        t_t_msgs[["target_a", "target_b"]].values.T, dtype=torch.long
    )
    return msgs, sups, t_t_msgs


def load_edges(
    s_s_labels_path: str,
    s_t_labels_path: str,
    t_t_labels_path: str,
    message: list[str],
    supervision: list[str],
    pretrain_source = False,
) -> tuple[torch.tensor, torch.tensor]:
    """Load messages and supervision edges in torch tensors"""
    s_s_edges = pd.read_parquet(s_s_labels_path)
    s_t_edges = pd.read_parquet(s_t_labels_path)
    t_t_edges = pd.read_parquet(t_t_labels_path)

    if pretrain_source:
        # train_loader and val_loader
        if message == ['message'] or message == ["message", "train"]:
            s_s_type = ["message", "train"]
            s_s_msgs = s_s_edges.query("subset.isin(@s_s_type)")
            s_s_msgs = torch.tensor(
                s_s_msgs[["source_a", "source_b"]].values.T, dtype=torch.long
            )

            s_s_type = ["valid"]
            s_s_sup = s_s_edges.query("subset.isin(@s_s_type)")
            s_s_sup = torch.tensor(
                s_s_sup[["source_a", "source_b"]].values.T, dtype=torch.long
            )
        else:
            # test_loader
            s_s_type = ["message", "train", "valid"]
            s_s_msgs = s_s_edges.query("subset.isin(@s_s_type)")
            s_s_msgs = torch.tensor(
                s_s_msgs[["source_a", "source_b"]].values.T, dtype=torch.long
            )

            s_s_type = ["test"]
            s_s_sup = s_s_edges.query("subset.isin(@s_s_type)")
            s_s_sup = torch.tensor(
                s_s_sup[["source_a", "source_b"]].values.T, dtype=torch.long
            )
        
        if message == ["message"]:
            t_t_msgs = t_t_edges.query("subset.isin(@supervision)")
        else:
            t_t_msgs = t_t_edges.query("subset.isin(@message)")
        t_t_msgs = torch.tensor(
            t_t_msgs[["target_a", "target_b"]].values.T, dtype=torch.long
        )

        if message == ["message"] or message == ["message", "train"]:
            s_t_type = ["message", "train", "valid"]
            s_t_msgs = s_t_edges.query("subset.isin(@s_t_type)")

        else:
            s_t_type = ["message", "train", "valid", "test"]
            s_t_msgs = s_t_edges.query("subset.isin(@s_t_type)")
        
        s_t_msgs = torch.tensor(s_t_msgs[["source", "target"]].values.T, dtype=torch.long)

        return s_s_msgs, s_s_sup, s_t_msgs, t_t_msgs 

    # set source target edges as message and supervision training edges
    msgs = s_t_edges.query("subset.isin(@message)")
    msgs = torch.tensor(msgs[["source", "target"]].values.T, dtype=torch.long)
    sups = s_t_edges.query("subset.isin(@supervision)")
    sups = torch.tensor(sups[["source", "target"]].values.T, dtype=torch.long)

    # only include message source/source and target_target edges
    if message == ["message"]:
        s_s_msgs = s_s_edges.query("subset.isin(@supervision)")
        t_t_msgs = t_t_edges.query("subset.isin(@supervision)")

    else:
        s_s_msgs = s_s_edges.query("subset.isin(@message)")
        t_t_msgs = t_t_edges.query("subset.isin(@message)")

    s_s_msgs = torch.tensor(
        s_s_msgs[["source_a", "source_b"]].values.T, dtype=torch.long
    )
    t_t_msgs = torch.tensor(
        t_t_msgs[["target_a", "target_b"]].values.T, dtype=torch.long
    )

    return msgs, sups, s_s_msgs, t_t_msgs


def load_bipartite_graph(
    source_path: str,
    target_path: str,
    labels_path: str,
    message: list[str],
    supervision: list[str],
) -> HeteroData:
    data = load_node_features(source_path, target_path)
    msgs, sups = load_bipartite_edges(labels_path, message, supervision)
    edge_label = torch.ones(sups.shape[1], dtype=torch.float)
    data["source", "binds", "target"].edge_index = msgs
    data["source", "binds", "target"].edge_label_index = sups
    data["source", "binds", "target"].edge_label = edge_label
    data = T.ToUndirected()(data).to(DEVICE, non_blocking=True)

    return data


def load_graph(
    source_path: str,
    target_path: str,
    s_s_labels_path: str,
    s_t_labels_path: str,
    t_t_labels_path: str,
    message: list[str],
    supervision: list[str],
    graph_type: str,
    pretrain_source=False,
) -> HeteroData:
    data = load_node_features(source_path, target_path)

    if pretrain_source:
        s_s_msgs, s_s_sups, s_t_msgs, t_t_msgs = load_edges(
            s_s_labels_path,
            s_t_labels_path,
            t_t_labels_path,
            message,
            supervision,
            pretrain_source = pretrain_source
        )
        data["target", "similar", "target"].edge_index = t_t_msgs
        data["source", "binds", "target"].edge_index = s_t_msgs

        edge_label = torch.ones(s_s_sups.shape[1], dtype=torch.float)
        data["source", "similar", "source"].edge_index = s_s_msgs
        data["source", "similar", "source"].edge_label_index = s_s_sups
        data["source", "similar", "source"].edge_label = edge_label

        data = T.ToUndirected()(data).to(DEVICE, non_blocking=True)

        return data


    if graph_type == "s_expanded":
        msgs, sups, s_s_msgs = load_s_expanded_edges(
            s_s_labels_path, s_t_labels_path, message, supervision
        )
        data["source", "similar", "source"].edge_index = s_s_msgs

    elif graph_type == "t_expanded":
        msgs, sups, t_t_msgs = load_t_expanded_edges(
            s_t_labels_path, t_t_labels_path, message, supervision
        )
        data["target", "similar", "target"].edge_index = t_t_msgs

    else:
        msgs, sups, s_s_msgs, t_t_msgs = load_edges(
            s_s_labels_path,
            s_t_labels_path,
            t_t_labels_path,
            message,
            supervision,
        )
        data["source", "similar", "source"].edge_index = s_s_msgs
        data["target", "similar", "target"].edge_index = t_t_msgs

    edge_label = torch.ones(sups.shape[1], dtype=torch.float)
    data["source", "binds", "target"].edge_index = msgs
    data["source", "binds", "target"].edge_label_index = sups
    data["source", "binds", "target"].edge_label = edge_label

    data = T.ToUndirected()(data).to(DEVICE, non_blocking=True)

    return data


def load_graph_helper(leave_out: str, tgt_type: str, graph_type: str, pretrain_source: bool):
    """
    Helper function to load the correct graph type based on
    datasplit, target profiles, and edge types. Eventually will take
    this out of hard code.
    """

    training = [["message"], ["train"]]
    validation = [["message", "train"], ["valid"]]
    testing = [["message", "train", "valid"], ["test"]]

    if graph_type == "bipartite":
        paths = [
            f"data/{graph_type}/{tgt_type}/source.parquet",
            f"data/{graph_type}/{tgt_type}/target.parquet",
            f"data/{graph_type}/{tgt_type}/{leave_out}/s_t_labels.parquet",
        ]
        train_data = load_bipartite_graph(*paths, *training)
        valid_data = load_bipartite_graph(*paths, *validation)
        test_data = load_bipartite_graph(*paths, *testing)

    else:
        paths = [
            f"data/{graph_type}/{tgt_type}/source.parquet",
            f"data/{graph_type}/{tgt_type}/target.parquet",
        ] + [
            f"data/{graph_type}/{tgt_type}/{leave_out}/{t}.parquet"
            for t in ("s_s_labels", "s_t_labels", "t_t_labels")
        ]
        train_data = load_graph(*paths, *training, graph_type, pretrain_source)
        valid_data = load_graph(*paths, *validation, graph_type, pretrain_source)
        test_data = load_graph(*paths, *testing, graph_type, pretrain_source)

        # print("train_data", train_data)
        # print("val_data", valid_data)
        # print("test_data", test_data)

    return train_data, valid_data, test_data


def get_loader(data: HeteroData, edges, leave_out, type: str, pretrain_source: bool) -> LinkNeighborLoader:
    if pretrain_source:
        edge_label_index = data["source", "similar", "source"].edge_label_index
        edge_label = data["source", "similar", "source"].edge_label
    else:
        edge_label_index = data["source", "binds", "target"].edge_label_index
        edge_label = data["source", "binds", "target"].edge_label

    if type == "train":
        bsz = 512
        shuffle = True
        ratio = 1
    else:
        bsz = 8192
        shuffle = False
        ratio = 10
    
    if pretrain_source:
        data_loader = LinkNeighborLoader(
            data=data,
            num_neighbors={key: [-1] * 4 for key in data.edge_types},
            edge_label_index=(("source", "similar", "source"), edge_label_index),
            edge_label=edge_label,
            transform=SampleNegatives(edges, leave_out, ratio, pretrain_source),
            subgraph_type="bidirectional",
            batch_size=bsz,
            shuffle=shuffle,
            filter_per_worker=True,
        )
    else:
        data_loader = LinkNeighborLoader(
            data=data,
            num_neighbors={key: [-1] * 4 for key in data.edge_types},
            edge_label_index=(("source", "binds", "target"), edge_label_index),
            edge_label=edge_label,
            transform=SampleNegatives(edges, leave_out, ratio),
            subgraph_type="bidirectional",
            batch_size=bsz,
            shuffle=shuffle,
            filter_per_worker=True,
        )
    return PrefetchLoader(loader=data_loader, device=DEVICE)


def get_loaders(
    leave_out: str, tgt_type: str, graph_type: str, pretrain_source: bool
) -> tuple[LinkNeighborLoader]:
    train_data, valid_data, test_data = load_graph_helper(
        leave_out, tgt_type, graph_type, pretrain_source
    )

    edges = get_all_st_edges(test_data, pretrain_source=pretrain_source)
    train_loader = get_loader(train_data, edges, leave_out, "train", pretrain_source)
    valid_loader = get_loader(valid_data, edges, leave_out, "valid", pretrain_source)
    test_loader = get_loader(test_data, edges, leave_out, "test", pretrain_source)

    return train_loader, valid_loader, test_loader
