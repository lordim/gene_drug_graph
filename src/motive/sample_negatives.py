import torch
import os
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


def sample_indices(scores, p, k, dev=None):
    N = len(scores)
    
    # Get top k scores and their indices
    top_k_scores, top_k_indices = torch.topk(scores, k)
    
    # Calculate the number of samples from top k and from all N
    p_k = int(p * k)
    remaining_k = k - p_k  # The remaining (1-p)*k

    # Randomly sample p*k from the top k indices
    top_k_sampled_indices = top_k_indices[torch.randperm(k)[:p_k]]
    
    # Randomly sample (1-p)*k from all N indices
    all_indices = torch.arange(N).to(dev)

    remaining_indices = all_indices[~torch.isin(all_indices, top_k_sampled_indices)]

    remaining_k_sampled_indices = remaining_indices[torch.randperm(len(remaining_indices))[:remaining_k]]
    
    # Combine the sampled indices to get k indices
    sampled_indices = torch.cat([top_k_sampled_indices, remaining_k_sampled_indices])
    
    return sampled_indices.detach().cpu().numpy()

def find_indices(reference, query):
    """
    get the indices of the query that are in reference.
    e.g.
    reference = torch.tensor([3, 10, 7])
    query = torch.tensor([7, 3, 3, 7, 10])
    find_indices(reference, query)
    >>> tensor([2, 0, 0, 2, 1])
    """
    return torch.max(reference[:, None] == query[None, :], axis=0).indices


def negative_sampling(source_ix, target_ix, pos_edges, size):
    """
    Negative sampling using GPU and batched impl.
    Create source_ix[i], target_ix[j] pairs that are not present in pos_edges.
    """
    # print("source_ix:", source_ix, len(source_ix))
    # print("target_ix:", target_ix, len(target_ix))

    size = size * 2
    neg_source_ix = torch.randperm(size) % len(source_ix)
    neg_source_ix = source_ix[neg_source_ix]
    neg_target_ix = torch.randperm(size) % len(target_ix)
    neg_target_ix = target_ix[neg_target_ix]
    samples = torch.stack([neg_source_ix, neg_target_ix]).T
    samples = torch.unique(samples, dim=0)

    y_true = torch.any(torch.all(samples[:, None] == pos_edges.T, axis=2), axis=1)
    samples = samples[~y_true]
    samples = samples[: size // 2].T

    # print("neg_samples shape:", samples.shape)
    # print("neg_samples:", samples)

    return samples

def negative_sampling_dynamic(source_ix, target_ix, pos_edges, size, 
                              data = None, gnn_model = None, explore_coeff = 2,
                              epoch = None, num_epochs = None):
    """
    Negative sampling using GPU and batched impl.
    Create source_ix[i], target_ix[j] pairs that are not present in pos_edges.
    Score the negative edges with the model and sample the top k = size edges.
    """
    DEVICE = torch.device(f"cuda:{os.getenv('GPU_DEVICE')}" if (os.getenv('GPU_DEVICE') != "cpu" and torch.cuda.is_available()) else "cpu")

    size = size * explore_coeff
    neg_source_ix = torch.randperm(size) % len(source_ix)
    neg_source_ix = source_ix[neg_source_ix]
    neg_target_ix = torch.randperm(size) % len(target_ix)
    neg_target_ix = target_ix[neg_target_ix]
    samples = torch.stack([neg_source_ix, neg_target_ix]).T
    samples = torch.unique(samples, dim=0)

    y_true = torch.any(torch.all(samples[:, None] == pos_edges.T, axis=2), axis=1)
    neg_samples = samples[~y_true]

    with torch.no_grad():
        neg_data = data.clone()
        neg_data["binds"].edge_label = torch.zeros(len(neg_samples)).to(DEVICE)
        neg_data["binds"].edge_label_index = neg_samples.T

        logits = gnn_model(neg_data)

        neg_data = neg_data.detach().cpu()
        
        # p = 1.0
        p = epoch / num_epochs
        sampled_indices = sample_indices(logits, p, size // explore_coeff, dev=DEVICE)

    neg_samples = neg_samples[sampled_indices].T

    return neg_samples


def select_nodes_to_sample(data, split):
    """Select nodes to build negative samples based on the split"""
    source_ix = data["binds"].edge_label_index[0]
    target_ix = data["binds"].edge_label_index[1]
    if split != "source":
        source_ix = torch.cat((source_ix, data["binds"].edge_index[0]))
    if split != "target":
        target_ix = torch.cat((target_ix, data["binds"].edge_index[1]))
    return source_ix.unique(), target_ix.unique()


class SampleNegatives(BaseTransform):
    def __init__(self, edges, split, ratio, 
                 all_data=None, gnn_model=None, epoch=None, num_epochs=None):

        self.device = torch.device(f"cuda:{os.getenv('GPU_DEVICE')}" if (os.getenv('GPU_DEVICE') != "cpu" and torch.cuda.is_available()) else "cpu")

        self.edges = torch.tensor(edges, device=self.device)
        self.split = split
        self.ratio = ratio

        self.all_data = all_data
        self.gnn_model = gnn_model

        self.epoch = epoch
        self.num_epochs = num_epochs

    def forward(self, data: HeteroData):
        data = data.to(self.device, non_blocking=True)

        num_pos = len(data["binds"].edge_label)
        # Select nodes
        subgraph_src, subgraph_tgt = select_nodes_to_sample(data, self.split)

        # map local (subgraph) edge indices to global indices
        global_src = data["source"].node_id[subgraph_src]
        global_tgt = data["target"].node_id[subgraph_tgt]

        size = num_pos * self.ratio
        if self.gnn_model:
            neg_edges = negative_sampling_dynamic(global_src, global_tgt, self.edges, size, 
                                                  self.all_data, self.gnn_model, explore_coeff=4,
                                                  epoch=self.epoch, num_epochs=self.num_epochs)
        else:
            neg_edges = negative_sampling(global_src, global_tgt, self.edges, size)

        # map global edge indices to local (subgraph) indices
        neg_src = find_indices(data["source"].node_id, neg_edges[0])
        neg_tgt = find_indices(data["target"].node_id, neg_edges[1])

        # concat current and new edges and labels
        neg_edges = torch.stack([neg_src, neg_tgt])
        new_edges = torch.cat((data["binds"].edge_label_index, neg_edges), axis=1)

        neg_label = torch.zeros(len(neg_src), device=self.device)
        new_label = torch.cat((data["binds"].edge_label, neg_label))

        # update data object
        data["binds"].edge_label = new_label
        data["binds"].edge_label_index = new_edges.contiguous()

        # print(data)
        # raise ValueError()
    
        return data