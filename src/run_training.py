import argparse
import os
import os.path
import torch

from motive import get_counts, get_loaders
from model import GraphSAGE_CP, GraphSAGE_Embs, MLP, Bilinear
from model import GraphTransformer_Embs, GraphTransformer_CP, GraphSAGE_OurFeat, GraphTransformer_OurFeat
from model import GraphAttention_OurFeat, GraphIsomorphism_OurFeat, RUM_Embs
# from train import DEVICE, train_loop
from train import train_loop
from utils.evaluate import save_metrics
from utils.utils import PathLocator

def initialize_model(locator, train_loader, model_name):
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
        
        elif initialization == "ourfeat":
            model = GraphSAGE_OurFeat(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )

        else:
            raise NotImplementedError(f"Initialization {initialization} not supported for GNN.")
    
    elif model_name == "gtn":
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
        
        elif initialization == "ourfeat":
            model = GraphTransformer_OurFeat(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )
        
        else:
            raise NotImplementedError("Only embs initialization is supported for GTN now.")

    elif model_name == "gat":
        initialization = locator.config["initialization"]
        if initialization == "ourfeat":
            model = GraphAttention_OurFeat(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )
    
    elif model_name == "gin":
        initialization = locator.config["initialization"]
        if initialization == "ourfeat":
            model = GraphIsomorphism_OurFeat(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )
    
    elif model_name == "rum":
        initialization = locator.config["initialization"]
        if initialization == "embs":
            model = RUM_Embs(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )

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
    
    return model  


def workflow(args, locator, num_epochs, tgt_type, graph_type, input_root_dir):
    DEVICE = torch.device(f"cuda:{os.getenv('GPU_DEVICE')}" if torch.cuda.is_available() else "cpu")

    leave_out = locator.config["data_split"]
    model_name = locator.config["model_name"]
    train_loader, val_loader, test_loader = get_loaders(leave_out, tgt_type, graph_type, input_root_dir)

    num_sources, num_targets, num_features = get_counts(train_loader.loader.data)

    # initialize model here:
    model = initialize_model(locator, train_loader, model_name)

    model = model.to(DEVICE)

    train_loader, val_loader, test_loader = get_loaders(leave_out, tgt_type, graph_type, model=model)
    results, test_scores, _ = train_loop(
        model, locator, train_loader, val_loader, test_loader, num_epochs,
        tgt_type, graph_type, input_root_dir,
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
    parser.add_argument("--gpu_device", type=str, dest="gpu_device", help="GPU device to use")

    parser.add_argument("--target_type", dest="target_type", default="orf")
    parser.add_argument("--graph_type", dest="graph_type", default="st_expanded")
    parser.add_argument("--input_root_dir", dest="input_root_dir", help="root directory for input/data")
    args = parser.parse_args()

    os.environ["GPU_DEVICE"] = args.gpu_device

    # imports HERE to avoid cuda visibility issues:
    # from motive import get_counts, get_loaders
    # from model import GraphSAGE_CP, GraphSAGE_Embs, MLP, Bilinear
    # from model import GraphTransformer_Embs, GraphTransformer_CP
    # from train import DEVICE, train_loop
    # from utils.evaluate import save_metrics
    # from utils.utils import PathLocator
    #-----------------------------------------------


    locator = PathLocator(args.config_path, args.output_path)
    if os.path.isfile(locator.test_results_path):
        print(f"{locator.test_results_path} exists. Skipping...")
        return

    workflow(
        args,
        locator,
        args.num_epochs,
        args.target_type,
        args.graph_type,
        args.input_root_dir,
    )


if __name__ == "__main__":
    main()
