import numpy as np
import pandas as pd
import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

SEED = 2024319

#TODO: change setting so that device can be set at start of training
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_indices(scores, p, k, dev=None):
    N = len(scores)

    # print("scores:", scores)
    # print("p:", p)
    # print("k:", k)
    
    # Get top k scores and their indices
    top_k_scores, top_k_indices = torch.topk(scores, k)
    
    # Calculate the number of samples from top k and from all N
    p_k = int(p * k)
    remaining_k = k - p_k  # The remaining (1-p)*k

    # Randomly sample p*k from the top k indices
    top_k_sampled_indices = top_k_indices[torch.randperm(k)[:p_k]]
    
    # Randomly sample (1-p)*k from all N indices
    all_indices = torch.arange(N).to(dev)

    # print("all_indices:", all_indices)
    # print("top_k indices:", top_k_indices)

    remaining_indices = all_indices[~torch.isin(all_indices, top_k_sampled_indices)]

    # print("remaining_indices:", remaining_indices)

    remaining_k_sampled_indices = remaining_indices[torch.randperm(len(remaining_indices))[:remaining_k]]
    
    # Combine the sampled indices to get k indices
    sampled_indices = torch.cat([top_k_sampled_indices, remaining_k_sampled_indices])
    
    return sampled_indices.detach().cpu().numpy()


class SampleNegatives(BaseTransform):
    def __init__(self, edges, datasplit, ratio=1, random_neg=False, gnn_model=None, 
                 epoch=None, num_epochs=None):
        self.edges = edges
        self.datasplit = datasplit
        self.ratio = ratio
        self.random_neg = random_neg

        self.gnn_model = gnn_model
        self.epoch = epoch
        self.num_epochs = num_epochs
        self.explore_coeff = 1

        self.device = torch.device(f"cuda:{os.getenv('GPU_DEVICE')}" if torch.cuda.is_available() else "cpu")

    def forward(self, data: HeteroData):
        num_pos = len(data["binds"].edge_label)

        if self.gnn_model:
            self.ratio = self.ratio * self.explore_coeff

        if self.datasplit == "source":
            subgraph_src = data["binds"].edge_label_index[0].unique()
            global_src = data["source"].node_id[subgraph_src]

            subgraph_tgt = torch.cat(
                (
                    data["binds"].edge_index[1].unique().cpu(),
                    data["binds"].edge_label_index[1].unique().cpu(),
                ),
                dim=0,
            ).unique()
            global_tgt = data["target"].node_id[subgraph_tgt]

        elif self.datasplit == "target":
            subgraph_src = torch.cat(
                (
                    data["binds"].edge_index[0].unique().cpu(),
                    data["binds"].edge_label_index[0].unique().cpu(),
                ),
                dim=0,
            ).unique()
            global_src = data["source"].node_id[subgraph_src]

            subgraph_tgt = data["binds"].edge_label_index[1].unique()
            global_tgt = data["target"].node_id[subgraph_tgt]

        elif self.datasplit == "random":
            subgraph_src = torch.cat(
                (
                    data["binds"].edge_index[0].unique().cpu(),
                    data["binds"].edge_label_index[0].unique().cpu(),
                ),
                dim=0,
            ).unique()
            global_src = data["source"].node_id[subgraph_src]

            subgraph_tgt = torch.cat(
                (
                    data["binds"].edge_index[1].unique().cpu(),
                    data["binds"].edge_label_index[1].unique().cpu(),
                ),
                dim=0,
            ).unique()
            global_tgt = data["target"].node_id[subgraph_tgt]

        subgraph_src = subgraph_src.cpu().numpy()
        global_src = global_src.cpu().numpy()
        subgraph_tgt = subgraph_tgt.cpu().numpy()
        global_tgt = global_tgt.cpu().numpy()

        pos_edges = pd.MultiIndex.from_arrays(self.edges)

        # 3 chances to sample negative edges
        if self.random_neg:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(SEED)
        for _ in range(3):
            rnd_srcs = rng.choice(global_src, size=(num_pos * self.ratio * 2))
            rnd_tgts = rng.choice(global_tgt, size=(num_pos * self.ratio * 2))

            rnd_pairs = np.stack((rnd_srcs, rnd_tgts))
            rnd_pairs = np.unique(rnd_pairs, axis=1)
            rnd_pairs = pd.MultiIndex.from_arrays(rnd_pairs)
            inter = rnd_pairs.intersection(pos_edges, sort=False)
            neg_pairs = rnd_pairs.difference(inter, sort=False)

            if len([*neg_pairs]) < (num_pos * self.ratio):
                continue
            neg_pairs = rng.choice([*neg_pairs], num_pos * self.ratio, replace=False).T
            break

        else:
            raise RuntimeError("Could not successfully sample negatives.")
        
        # print("Neg pairs before adjustment:", neg_pairs.shape)
        # ---------------HERE----------------
        # self.gnn_model = None
        if self.gnn_model:
            # print(f"Epoch {self.epoch}, SHOULD BE HERE!!!!")
            with torch.no_grad():
                neg_data = data.clone()
                # print(neg_data)
                neg_data["binds"].edge_label = torch.zeros(num_pos * self.ratio * self.explore_coeff).to(self.device)

                # build dictionaries to map global edge indices to local (subgraph) indices
                source_map = dict(zip(pd.Series(global_src), pd.Series(subgraph_src)))
                target_map = dict(zip(pd.Series(global_tgt), pd.Series(subgraph_tgt)))

                neg_edges_srcs = pd.Series(neg_pairs[0]).map(source_map).values
                neg_edges_tgts = pd.Series(neg_pairs[1]).map(target_map).values

                neg_data["binds"].edge_label_index = torch.Tensor(np.array([neg_edges_srcs, neg_edges_tgts])).type(torch.int32).to(self.device)

                logits = self.gnn_model(neg_data)

                # del self.gnn_model
                
                neg_data = neg_data.detach().cpu()

                p = self.epoch / self.num_epochs
                sampled_indices = sample_indices(logits, p, num_pos * self.ratio, dev=self.device)
                neg_pairs = neg_pairs[:, sampled_indices]

            # print("Neg pairs after adjustment:", neg_pairs.shape)
            
            # print("num_pos:", num_pos)
            # print("neg_pairs:", neg_pairs.shape)

            # sort the logits and get the top k
            # sample with probs p and q accoridng to our algorithm specified
            # with the sample's indices, update neg_pairs

            # raise ValueError("stop here")
        # -----------------------------------

        # build dictionaries to map global edge indices to local (subgraph) indices
        source_map = dict(zip(pd.Series(global_src), pd.Series(subgraph_src)))
        target_map = dict(zip(pd.Series(global_tgt), pd.Series(subgraph_tgt)))

        neg_edges_srcs = pd.Series(neg_pairs[0]).map(source_map).values
        neg_edges_tgts = pd.Series(neg_pairs[1]).map(target_map).values

        new_labels = torch.cat(
            (
                data["binds"].edge_label.cpu(),
                torch.Tensor(np.zeros(num_pos * self.ratio)),
            )
        ).to(self.device)#.to(DEVICE)
        new_edges = (
            torch.cat(
                (
                    data["binds"].edge_label_index.cpu(),
                    torch.Tensor(np.array([neg_edges_srcs, neg_edges_tgts])),
                ),
                axis=1,
            )
            .type(torch.int32)
            .to(self.device)#.to(DEVICE)
        )

        data["binds"].edge_label = new_labels
        data["binds"].edge_label_index = new_edges

        return data
