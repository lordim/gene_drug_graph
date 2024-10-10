import argparse
import os.path

from motive import get_counts, get_loaders
from model import GraphSAGE_CP, GraphSAGE_Embs, MLP, Bilinear
from model import GraphTransformer_Embs, GraphTransformer_CP
from train import DEVICE, train_loop
from utils.evaluate import save_metrics
from utils.utils import PathLocator


def workflow(locator, num_epochs, tgt_type, graph_type, input_root_dir):
    leave_out = locator.config["data_split"]
    model_name = locator.config["model_name"]
    train_loader, val_loader, test_loader = get_loaders(leave_out, tgt_type, graph_type, input_root_dir)

    num_sources, num_targets, num_features = get_counts(train_loader.loader.data)

    if model_name == "gnn":
        initialization = locator.config["initialization"]
        if initialization == "cp":
            model = GraphSAGE_CP(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )
        elif initialization == "embs":
            model = GraphSAGE_Embs(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )
    
    if model_name == "gtn":
        initialization = locator.config["initialization"]
        if initialization == "cp":
            model = GraphTransformer_CP(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )
        
        elif initialization == "embs":
            model = GraphTransformer_Embs(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )
        
        else:
            raise NotImplementedError("Only embs initialization is supported for GTN now.")

    elif model_name == "mlp":
        model = MLP(
            num_features["source"],
            num_features["target"],
            hidden_size=int(locator.config["hidden_channels"]),
        )

    elif model_name == "bilinear":
        model = Bilinear(
            num_features["source"],
            num_features["target"],
        )

    model = model.to(DEVICE)
    results, test_scores, _ = train_loop(
        model, locator, train_loader, val_loader, test_loader, num_epochs
    )
    save_metrics(test_scores, locator.test_metrics_path)
    results.to_parquet(locator.test_results_path)
    print(test_scores)


def main():
    """Parse input params"""
    parser = argparse.ArgumentParser(
        description=("Train GNN with this config file"),
    )

    parser.add_argument("config_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1000)

    parser.add_argument("--target_type", dest="target_type", default="orf")
    parser.add_argument("--graph_type", dest="graph_type", default="st_expanded")
    parser.add_argument("--input_root_dir", dest="input_root_dir", default="root directory for input/data")
    args = parser.parse_args()

    locator = PathLocator(args.config_path, args.output_path)
    if os.path.isfile(locator.test_results_path):
        print(f"{locator.test_results_path} exists. Skipping...")
        return
    workflow(
        locator,
        args.num_epochs,
        args.target_type,
        args.graph_type,
        args.input_root_dir,
    )


if __name__ == "__main__":
    main()