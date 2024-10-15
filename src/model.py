import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, TransformerConv, GATv2Conv, GINConv, MLP as tMLP
from torch_geometric.nn.norm import HeteroLayerNorm


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)

        # self.layernorm = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Define a 2-layer GNN computation graph.
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = h1 + h2
        return h3

class GTN(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()

        self.conv1 = TransformerConv(hidden_channels, hidden_channels, heads=num_heads, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=num_heads, concat=False)

        self.layernorm = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.layernorm(x)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()

        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=num_heads, add_self_loops=False, concat=False)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=num_heads, add_self_loops=False, concat=False)

        self.layernorm = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.layernorm(x)
        x = self.conv2(x, edge_index)
        return x

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.mlp1 = tMLP(
            [hidden_channels, hidden_channels, hidden_channels],
            act="leaky_relu",
            norm="layernorm",
            dropout=0.1,
        )
        self.mlp2 = tMLP(
            [hidden_channels, hidden_channels, hidden_channels],
            act="leaky_relu",
            norm="layernorm",
            dropout=0.1,
        )

        self.conv1 = GINConv(self.mlp1, train_eps=True)
        self.conv2 = GINConv(self.mlp2, train_eps=True)

        # self.batchnorm = torch.nn.BatchNorm1d(hidden_channels)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h1 = self.conv1(x, edge_index)
        h1 = F.normalize(h1, dim=-1)
        h1 = F.leaky_relu(h1)

        h2 = self.conv2(h1, edge_index)
        h2 = F.normalize(h2, dim=-1)
        # h3 = h1 + h2
        return h2


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

#------------------------- RUM MODELS HERE ----------------------------

from rum_model import RUMModel
class RUM_Embs(torch.nn.Module):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__()

        # embedding matrices for sources and targets:
        self.source_emb = torch.nn.Embedding(num_source_nodes, hidden_channels)
        self.target_emb = torch.nn.Embedding(num_target_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.rum = RUMModel(
            in_features=hidden_channels,
            out_features=hidden_channels,
            hidden_features=hidden_channels,
            depth=2,
        )

        # Convert GNN model into a heterogeneous variant:
        metadata = data.metadata()
        self.rum = to_hetero(self.rum, metadata=metadata)

        # Instantiate one of the classifier classes
        self.classifier = Classifier()
    
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "source": self.source_emb(data["source"].node_id),
            "target": self.target_emb(data["target"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        if self.training:
            x_dict, c_loss = self.rum(x_dict, data.edge_index_dict)
        else:
            x_dict = self.rum(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["source"],
            x_dict["target"],
            data["source", "binds", "target"].edge_label_index,
        )
        return pred

#------------------------- GIN MODELS HERE ----------------------------
class GraphIsomorphism_Embs(torch.nn.Module):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__()

        # embedding matrices for sources and targets:
        self.source_emb = torch.nn.Embedding(num_source_nodes, hidden_channels)
        self.target_emb = torch.nn.Embedding(num_target_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gin = GIN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        metadata = data.metadata()
        self.gin = to_hetero(self.gin, metadata=metadata)

        # Instantiate one of the classifier classes
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "source": self.source_emb(data["source"].node_id),
            "target": self.target_emb(data["target"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gin(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["source"],
            x_dict["target"],
            data["source", "binds", "target"].edge_label_index,
        )
        return pred

class GraphIsomorphism_OurFeat(GraphIsomorphism_Embs):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__(hidden_channels, num_source_nodes, num_target_nodes, data)
        src_weights = data["source"].x
        # tgt_weights = data["target"].x
        source_size = data["source"].x.shape[1]
        # target_size = data["target"].x.shape[1]

        self.source_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_source_nodes, source_size, _weight=src_weights, _freeze=True
            ),
            torch.nn.Linear(source_size, hidden_channels),
            torch.nn.LeakyReLU(),
        )

        # self.target_emb = torch.nn.Sequential(
        #     torch.nn.Embedding(
        #         num_target_nodes, target_size, _weight=tgt_weights, _freeze=True
        #     ),
        #     torch.nn.Linear(target_size, hidden_channels),
        #     torch.nn.ReLU(),
        # )

#------------------------- GAT MODELS HERE ----------------------------
class GraphAttention_Embs(torch.nn.Module):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__()

        # embedding matrices for sources and targets:
        self.source_emb = torch.nn.Embedding(num_source_nodes, hidden_channels)
        self.target_emb = torch.nn.Embedding(num_target_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gat = GAT(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        metadata = data.metadata()
        self.gat = to_hetero(self.gat, metadata=metadata)

        # Instantiate one of the classifier classes
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "source": self.source_emb(data["source"].node_id),
            "target": self.target_emb(data["target"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gat(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["source"],
            x_dict["target"],
            data["source", "binds", "target"].edge_label_index,
        )
        return pred

# Child of our GAT model that initializes embedding weights with
# our own features: ADMET feature for source, CURRENTLY random for target
class GraphAttention_OurFeat(GraphAttention_Embs):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__(hidden_channels, num_source_nodes, num_target_nodes, data)
        src_weights = data["source"].x
        # tgt_weights = data["target"].x
        source_size = data["source"].x.shape[1]
        # target_size = data["target"].x.shape[1]

        self.source_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_source_nodes, source_size, _weight=src_weights, _freeze=True
            ),
            torch.nn.Linear(source_size, hidden_channels),
            torch.nn.LeakyReLU(),
        )

        # self.target_emb = torch.nn.Sequential(
        #     torch.nn.Embedding(
        #         num_target_nodes, target_size, _weight=tgt_weights, _freeze=True
        #     ),
        #     torch.nn.Linear(target_size, hidden_channels),
        #     torch.nn.ReLU(),
        # )


#------------------------- Transformer MODELS HERE -------------------------

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
            torch.nn.LeakyReLU(),
        )

        self.target_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_target_nodes, target_size, _weight=tgt_weights, _freeze=True
            ),
            torch.nn.Linear(target_size, hidden_channels),
            torch.nn.LeakyReLU(),
        )

# Child of our GTN model that initializes embedding weights with
# our own features: ADMET feature for source, CURRENTLY random for target
class GraphTransformer_OurFeat(GraphTransformer_Embs):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__(hidden_channels, num_source_nodes, num_target_nodes, data)
        src_weights = data["source"].x
        # tgt_weights = data["target"].x
        source_size = data["source"].x.shape[1]
        # target_size = data["target"].x.shape[1]

        self.source_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_source_nodes, source_size, _weight=src_weights, _freeze=True
            ),
            torch.nn.Linear(source_size, hidden_channels),
            torch.nn.LeakyReLU(),
        )

        # self.target_emb = torch.nn.Sequential(
        #     torch.nn.Embedding(
        #         num_target_nodes, target_size, _weight=tgt_weights, _freeze=True
        #     ),
        #     torch.nn.Linear(target_size, hidden_channels),
        #     torch.nn.ReLU(),
        # )

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
            torch.nn.LeakyReLU(),
        )

        self.target_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_target_nodes, target_size, _weight=tgt_weights, _freeze=True
            ),
            torch.nn.Linear(target_size, hidden_channels),
            torch.nn.LeakyReLU(),
        )

# Child of our GNN model that initializes embedding weights with
# our own features: ADMET feature for source, CURRENTLY random for target
class GraphSAGE_OurFeat(GraphSAGE_Embs):
    def __init__(self, hidden_channels, num_source_nodes, num_target_nodes, data):
        super().__init__(hidden_channels, num_source_nodes, num_target_nodes, data)
        src_weights = data["source"].x
        # tgt_weights = data["target"].x
        source_size = data["source"].x.shape[1]
        # target_size = data["target"].x.shape[1]

        self.source_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_source_nodes, source_size, _weight=src_weights, _freeze=True
            ),
            torch.nn.Linear(source_size, hidden_channels),
            torch.nn.LeakyReLU(),
        )

        # self.target_emb = torch.nn.Sequential(
        #     torch.nn.Embedding(
        #         num_target_nodes, target_size, _weight=tgt_weights, _freeze=True
        #     ),
        #     torch.nn.Linear(target_size, hidden_channels),
        #     torch.nn.ReLU(),
        # )

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
        h_source = F.leaky_relu(self.dense_source(x_source))
        h_target = F.leaky_relu(self.dense_target(x_target))
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
