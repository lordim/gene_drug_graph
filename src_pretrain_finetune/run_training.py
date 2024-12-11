import argparse
import os.path

from model import create_model
from motive import get_loaders
from train import DEVICE, run_test, train_loop
from utils.evaluate import save_metrics
from utils.utils import PathLocator

import torch


def workflow(locator, num_epochs, tgt_type, graph_type, pretrain_source=False, pretrain_target=False, eval_test=False, pretrained_path = None):
    leave_out = locator.config["data_split"]
    train_loader, val_loader, test_loader = get_loaders(leave_out, tgt_type, graph_type, pretrain_source, pretrain_target)
    train_data = train_loader.loader.data
    model = create_model(locator, train_data, pretrain_source, pretrain_target).to(DEVICE)

    # Load pretrained_model
    if pretrained_path is not None: 

        pretrained_state_dict = torch.load(pretrained_path)
        model_state_dict = model.state_dict()

        if locator.config["model_name"] == "gin":
            s_s_weights = ["source_emb.0.weight", "source_emb.1.weight", "source_emb.1.bias"]

            for weight in s_s_weights:
                model_state_dict[weight] = pretrained_state_dict["model_state_dict"][weight]
            for weight in list(model_state_dict.keys()):
                if "source__similar__source" in weight:
                    model_state_dict[weight] = pretrained_state_dict["model_state_dict"][weight]
            model.load_state_dict(model_state_dict)

        else:
            s_s_weights = ["source_emb.0.weight", "source_emb.1.weight", "source_emb.1.bias",  
                        "gnn.conv1.source__similar__source.lin_l.weight", 
                        'gnn.conv1.source__similar__source.lin_l.bias', 'gnn.conv1.source__similar__source.lin_r.weight', 
                        'gnn.conv2.source__similar__source.lin_l.weight', 'gnn.conv2.source__similar__source.lin_l.bias', 
                        'gnn.conv2.source__similar__source.lin_r.weight']

            for weight in s_s_weights:
                model_state_dict[weight] = pretrained_state_dict["model_state_dict"][weight]

            model.load_state_dict(model_state_dict)
    best_th = train_loop(model, locator, train_loader, val_loader, num_epochs, pretrain_source=pretrain_source, pretrain_target=pretrain_target)
    if eval_test:
        results, test_scores = run_test(model, test_loader, best_th, pretrain_source = pretrain_source, pretrain_target=pretrain_target)
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
    parser.add_argument("--pretrain_source", type=bool, default=False)
    parser.add_argument("--pretrain_target", type=bool, default=False)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1000)

    parser.add_argument("--target_type", dest="target_type", default="orf")
    parser.add_argument("--graph_type", dest="graph_type", default="st_expanded")

    args = parser.parse_args()

    locator = PathLocator(args.config_path, args.output_path)
    if os.path.isfile(locator.test_results_path):
        print(f"{locator.test_results_path} exists. Skipping...")
        return
    workflow(
        locator, args.num_epochs, args.target_type, args.graph_type, args.pretrain_source, args.pretrain_target, eval_test=True, pretrained_path = args.pretrained_path
    )


if __name__ == "__main__":
    main()
