import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, TransformerConv


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Define a 2-layer GNN computation graph.
        h1 = F.leaky_relu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = h1 + h2
        return h3

class GTN(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()

        self.conv1 = TransformerConv(hidden_channels, hidden_channels, heads=num_heads, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=num_heads, concat=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(
        self, x_source: Tensor, x_target: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_source = x_source[edge_label_index[0]]
        edge_feat_target = x_target[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_source * edge_feat_target).sum(dim=-1)

#------------------------- OUR MODELS HERE -------------------------

class GraphTransformer_Embs(torch.nn.Module):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__()

        # embedding matrices for sources and targets:
        self.source_emb = torch.nn.Embedding(num_source_nodes, hidden_channels)
        self.target_emb = torch.nn.Embedding(num_target_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gtn = GTN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        metadata = data.metadata()
        self.gtn = to_hetero(self.gtn, metadata=metadata)

        # Instantiate one of the classifier classes
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "source": self.source_emb(data["source"].node_id),
            "target": self.target_emb(data["target"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gtn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["source"],
            x_dict["target"],
            data["source", "binds", "target"].edge_label_index,
        )
        return pred

# Child of our GTN model that initializes embedding weights with
# cp features but freezes embeddings throughout training
class GraphTransformer_CP(GraphTransformer_Embs):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__(hidden_channels, num_source_nodes, num_target_nodes, data)
        src_weights = data["source"].x
        tgt_weights = data["target"].x
        source_size = data["source"].x.shape[1]
        target_size = data["target"].x.shape[1]

        self.source_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_source_nodes, source_size, _weight=src_weights, _freeze=True
            ),
            torch.nn.Linear(source_size, hidden_channels),
            torch.nn.ReLU(),
        )

        self.target_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_target_nodes, target_size, _weight=tgt_weights, _freeze=True
            ),
            torch.nn.Linear(target_size, hidden_channels),
            torch.nn.ReLU(),
        )

#-------------------------------------------------------------------



class GraphSAGE_Embs(torch.nn.Module):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__()

        # embedding matrices for sources and targets:
        self.source_emb = torch.nn.Embedding(num_source_nodes, hidden_channels)
        self.target_emb = torch.nn.Embedding(num_target_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        metadata = data.metadata()
        self.gnn = to_hetero(self.gnn, metadata=metadata)

        # Instantiate one of the classifier classes
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "source": self.source_emb(data["source"].node_id),
            "target": self.target_emb(data["target"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["source"],
            x_dict["target"],
            data["source", "binds", "target"].edge_label_index,
        )
        return pred


# Child of our GNN model that initializes embedding weights with
# cp features but freezes embeddings throughout training
class GraphSAGE_CP(GraphSAGE_Embs):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__(hidden_channels, num_source_nodes, num_target_nodes, data)
        src_weights = data["source"].x
        tgt_weights = data["target"].x
        source_size = data["source"].x.shape[1]
        target_size = data["target"].x.shape[1]

        self.source_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_source_nodes, source_size, _weight=src_weights, _freeze=True
            ),
            torch.nn.Linear(source_size, hidden_channels),
            torch.nn.ReLU(),
        )

        self.target_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_target_nodes, target_size, _weight=tgt_weights, _freeze=True
            ),
            torch.nn.Linear(target_size, hidden_channels),
            torch.nn.ReLU(),
        )


class MLP(torch.nn.Module):
    def __init__(self, source_size, target_size, hidden_size):
        super().__init__()

        self.dense_source = torch.nn.Linear(source_size, hidden_size)
        self.dense_target = torch.nn.Linear(target_size, hidden_size)
        source_size = target_size = hidden_size
        self.bilinear = torch.nn.Bilinear(source_size, target_size, 1)

    def forward(self, data: HeteroData) -> Tensor:
        source_ix = data["binds"]["edge_label_index"][0]
        target_ix = data["binds"]["edge_label_index"][1]
        x_source = data["source"].x[source_ix]
        x_target = data["target"].x[target_ix]
        h_source = F.relu(self.dense_source(x_source))
        h_target = F.relu(self.dense_target(x_target))
        logits = self.bilinear(h_source, h_target)
        return torch.squeeze(logits)


class Bilinear(torch.nn.Module):
    def __init__(self, source_size, target_size):
        super().__init__()
        self.bilinear = torch.nn.Bilinear(source_size, target_size, 1)

    def forward(self, data: HeteroData) -> Tensor:
        source_ix = data["binds"]["edge_label_index"][0]
        target_ix = data["binds"]["edge_label_index"][1]
        x_source = data["source"].x[source_ix]
        x_target = data["target"].x[target_ix]
        logits = self.bilinear(x_source, x_target)
        return torch.squeeze(logits)
