import argparse
import os.path

from model import create_model
from motive import get_loaders
from train import DEVICE, run_test, train_loop
from utils.evaluate import save_metrics
from utils.utils import PathLocator

import torch


def workflow(locator, num_epochs, tgt_type, graph_type, pretrain_source=False, eval_test=False, pretrained_path = None):
    leave_out = locator.config["data_split"]
    train_loader, val_loader, test_loader = get_loaders(leave_out, tgt_type, graph_type, pretrain_source)
    train_data = train_loader.loader.data
    model = create_model(locator, train_data, pretrain_source).to(DEVICE)

    # print("model", model)

    # Load pretrained_model
    if pretrained_path is not None: 
        best_params = torch.load(pretrained_path)
        # print("best_params", best_params)
       
        model.load_state_dict(best_params["model_state_dict"])

        ## only use s-s weights from the pretrained path. 
        # pretrained_state_dict = torch.load(pretrained_path)
        # model_state_dict = model.state_dict()
        # model_state_dict[] = 

    best_th = train_loop(model, locator, train_loader, val_loader, num_epochs, pretrain_source=pretrain_source)
    if eval_test:
        results, test_scores = run_test(model, test_loader, best_th, pretrain_source = pretrain_source)
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
        locator, args.num_epochs, args.target_type, args.graph_type, args.pretrain_source, eval_test=True, pretrained_path = args.pretrained_path
    )


if __name__ == "__main__":
    main()
