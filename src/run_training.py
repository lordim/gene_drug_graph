import argparse
import os
import torch
import wandb

from motive import get_counts, get_loaders
# from model import GraphSAGE_CP, GraphSAGE_Embs, MLP, Bilinear
# from model import GraphTransformer_Embs, GraphTransformer_CP, GraphSAGE_OurFeat, GraphTransformer_OurFeat
# from model import GraphAttention_OurFeat, GraphIsomorphism_OurFeat, RUM_Embs
from model import GNN_CP, GNN_Embs, GNN_OurFeat, MLP, Bilinear, create_model
# from train import DEVICE, train_loop
from train import train_loop, run_test
from utils.evaluate import save_metrics
from utils.utils import PathLocator



def workflow(args, locator, num_epochs, tgt_type, graph_type, input_root_dir, eval_test=False):
    # DEVICE = torch.device(f"cuda:{os.getenv('GPU_DEVICE')}" if (os.getenv('GPU_DEVICE') != "cpu" and torch.cuda.is_available()) else "cpu")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    leave_out = locator.config["data_split"]
    # model_name = locator.config["model_name"]
    train_loader, val_loader, test_loader = get_loaders(args, leave_out, tgt_type, graph_type, input_root_dir,
                                                        init_feature=locator.config["initialization"])
    train_data = train_loader.loader.data

    # num_sources, num_targets, num_features = get_counts(train_loader.loader.data)
    # initialize model here:
    model = create_model(locator, train_data, args.use_wandb).to(DEVICE)

    # model = initialize_model(locator, train_loader, model_name)

    # model = model.to(DEVICE)
    best_th = train_loop(
        args, model, locator, train_loader, val_loader, test_loader, num_epochs,
        tgt_type, graph_type, input_root_dir,
    )
    if eval_test:
        results, test_scores = run_test(model, test_loader, best_th)
        save_metrics(test_scores, locator.test_metrics_path)
        results.to_parquet(locator.test_results_path)
        print(test_scores)

#test


def main():
    """Parse input params"""
    parser = argparse.ArgumentParser(
        description=("Train GNN with this config file"),
    )

    parser.add_argument("config_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=500)
    parser.add_argument("--gpu_device", type=str, dest="gpu_device", help="GPU device to use")

    # WANDB ARGS:
    parser.add_argument("--wandb_project", dest="wandb_project", default="huggingface")
    parser.add_argument("--wandb_entity", dest="wandb_entity", default="ziyixu686")
    parser.add_argument("--wandb_group", dest="wandb_group", default="test")
    parser.add_argument("--wandb_runid", dest="wandb_runid", default="test")
    parser.add_argument("--wandb_output_dir", dest="wandb_output_dir", default="~/tmp")
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true", default=False)

    # TRAINING ARGS:
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sample_neg_every_epoch", dest="sample_neg_every_epoch", action="store_true", default=False)
    parser.add_argument("--train_neg_ratio", dest="train_neg_ratio", type=int, default=1)
    parser.add_argument("--neg_explore_ratio", dest="neg_explore_ratio", type=int, default=1)
    parser.add_argument("--random_neg_ratio", dest="random_neg_ratio", action="store_true", default=False)

    parser.add_argument("--target_type", dest="target_type", default="orf")
    parser.add_argument("--graph_type", dest="graph_type", default="st_expanded")
    parser.add_argument("--input_root_dir", dest="input_root_dir", help="root directory for input/data")
    args = parser.parse_args()

    # os.environ["GPU_DEVICE"] = args.gpu_device

    locator = PathLocator(args.config_path, args.output_path)
    if os.path.isfile(locator.test_results_path):
        print(f"{locator.test_results_path} exists. Skipping...")
        return
    
    # if args.use_wandb:
    wandb_output_dir = os.path.join(args.wandb_output_dir, args.wandb_runid)
    if not os.path.exists(wandb_output_dir):
        os.makedirs(wandb_output_dir)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            dir=wandb_output_dir,
            name=args.wandb_runid,
            id=args.wandb_runid,
            config=locator.config,
        )

    workflow(
        args,
        locator,
        args.num_epochs,
        args.target_type,
        args.graph_type,
        args.input_root_dir,
        eval_test=True,
    )

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
