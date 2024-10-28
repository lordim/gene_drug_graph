import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
import copy


# SHOULD WE UNSET THE SEED? 
SEED = 2024319

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SampleNegatives(BaseTransform):
    def __init__(self, edges, datasplit, ratio=1, model=None, is_train =False):
        self.edges = edges
        self.datasplit = datasplit
        self.ratio = ratio
        self.model=model
        self.is_train = is_train

    def forward(self, data: HeteroData):
        num_pos = len(data["binds"].edge_label)

        # with the command I am using, self.datasplit = "source"

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
            
            ######## start of modifications ######## 
            # Obtain the negative pairs using the idea: 
            # given the candidate negative edges, pass them through the model to obtain logits. 
            # Retain negative edges with top logits. 
            # number_to_retain = num_pos * self.ratio    (same as in original implementations)
            
            original_edge_label_index = data["binds"].edge_label_index
            if self.is_train:
                
                neg_pairs_unfiltered = [list(tup) for tup in neg_pairs]
                neg_pairs_unfiltered = np.array(neg_pairs_unfiltered).T   # (2,256)

                source_map = dict(zip(pd.Series(global_src), pd.Series(subgraph_src)))
                target_map = dict(zip(pd.Series(global_tgt), pd.Series(subgraph_tgt)))

                neg_edges_srcs = pd.Series(neg_pairs_unfiltered[0]).map(source_map).values
                neg_edges_tgts = pd.Series(neg_pairs_unfiltered[1]).map(target_map).values

                neg_edges = (
                    torch.Tensor(np.array([neg_edges_srcs, neg_edges_tgts]))
                    .type(torch.int32)
                    .to(DEVICE)
                )

                data["binds"].edge_label_index = neg_edges

                logits = self.model(data)
                
                number_to_retain = num_pos * self.ratio
                # sort the logits in decreasing order. Get the sorted indices.  
                sorted_logits_indices = np.argsort(logits.detach().cpu().numpy())[::-1]

                # keep top k indices, k = number_to_retain
                top_indices_retain = sorted_logits_indices[:number_to_retain]

                neg_pairs_source = neg_pairs_unfiltered[0][top_indices_retain]
                neg_pairs_target = neg_pairs_unfiltered[1][top_indices_retain]
    
                neg_pairs= np.array([neg_pairs_source, neg_pairs_target]) # shape (2,128)
            
            else:
                neg_pairs = rng.choice([*neg_pairs], num_pos * self.ratio, replace=False).T

            #### End of modifications ######
            break

        else:
            print("line 143 is called")
            raise RuntimeError("Could not successfully sample negatives.")

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
        ).to(DEVICE)
        new_edges = (
            torch.cat(
                (
                    original_edge_label_index.cpu(),
                    torch.Tensor(np.array([neg_edges_srcs, neg_edges_tgts])),
                ),
                axis=1,
            )
            .type(torch.int32)
            .to(DEVICE)
        )

        data["binds"].edge_label = new_labels
        data["binds"].edge_label_index = new_edges

        return data
