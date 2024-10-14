import argparse
import os
import json
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from plot.plot_exploration import contour
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.autonotebook import tqdm

from motive import get_counts, get_loaders
from model import MLP, Bilinear, GraphSAGE_CP, GraphSAGE_Embs
from model import GraphTransformer_CP, GraphTransformer_Embs
from model import GraphAttention_OurFeat
from train import SEED, run_eval_epoch, run_train_epoch #, DEVICE
from utils.evaluate import Evaluator, get_best_th
from utils.utils import PathLocator


def generate_parameters(output_dir): #(num_opts: int):
    config_search = []
    rng = np.random.default_rng(SEED)

    for hidden_channels in [64, 128, 256]:
        for learning_rate in [3e-3, 3e-4, 3e-5]:
            for weight_decay in [0.01, 0.05]:
                config_search.append((hidden_channels, learning_rate, weight_decay))
    #---------------------------------------------------------
    # for i in range(num_opts):
    #     hidden_channels = rng.choice([64, 128, 256])
    #     learning_rate = 10.0 ** rng.uniform(-6, -2)
    #     weight_decay = 10.0 ** rng.uniform(-5, 0)

    #     config_search.append((hidden_channels, learning_rate, weight_decay))
    #---------------------------------------------------------

    config_search_df = pd.DataFrame(
        config_search, columns=["hidden_channels", "learning_rate", "weight_decay"]
    )

    config_search_df.to_csv(os.path.join(output_dir, "optimize_configs.csv"), index=False)
    return config_search_df


def optimize_train_loop(locator, num_epochs, tgt_type, graph_type, input_root_dir, device):
    leave_out = locator.config["data_split"]
    model_name = locator.config["model_name"]

    train_loader, val_loader, _ = get_loaders(leave_out, tgt_type, graph_type, input_root_dir)
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
            model = GraphSAGE_Embs(
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
        
        else:
            raise NotImplementedError(f"Initialization {initialization} not supported for GTN.")
    
    elif model_name == "gat":
        initialization = locator.config["initialization"]
        if initialization == "ourfeat":
            model = GraphAttention_OurFeat(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                train_loader.loader.data,
            )
        
        else:
            raise NotImplementedError(f"Initialization {initialization} not supported for GAT.")

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
    
    else:
        raise NotImplementedError(f"Model {model_name} not found!")

    model = model.to(device)
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=locator.config["learning_rate"],
        weight_decay=locator.config["weight_decay"],
    )
    writer = SummaryWriter(
        log_dir=locator.summary_path, comment=locator.config["model_name"]
    )

    if num_epochs < 1:
        raise ValueError("Invalid number of epochs")
    ground_truth, logits = None, None
    best_metric = 0
    criteria = "F1"
    for epoch in tqdm(range(1, num_epochs + 1)):
        run_train_epoch(model, train_loader, optimizer, writer, epoch)
        curr_gt, curr_logits, val_metrics = run_eval_epoch(
            model, val_loader, writer, epoch
        )
        if val_metrics[criteria] > best_metric:
            best_metric = val_metrics[criteria]
            ground_truth, logits = curr_gt, curr_logits
            best_th = get_best_th(ground_truth, logits)
            state = dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                best_th=best_th,
            )
            torch.save(state, locator.model_path)
    writer.add_hparams(locator.config, {f"best_{criteria}": best_metric}, run_name="./")
    return ground_truth, logits, best_metric


def main():
    """Parse input params"""
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str)
    parser.add_argument("--i", dest="initialization", type=str)
    # parser.add_argument("num_search", type=int)
    parser.add_argument("data_split", type=str)
    parser.add_argument(
        "--generate_new_params", dest="generate_new", action="store_true", default=False
    )
    parser.add_argument("--input_root_dir", dest="input_root_dir", help="root directory for input/data")
    parser.add_argument("output_path", type=str)
    parser.add_argument("--gpu_device", type=str, dest="gpu_device", help="GPU device to use")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=1000)
    parser.add_argument("--target_type", dest="target_type", default="orf")
    parser.add_argument("--graph_type", dest="graph_type", default="st_expanded")

    args = parser.parse_args()

    os.environ["GPU_DEVICE"] = args.gpu_device

    # generate new random search parameters
    if args.generate_new:
        # configs = generate_parameters(args.num_search)
        configs = generate_parameters(args.output_path)

    # or load previously generated parameters
    else:
        configs = pd.read_csv("inputs/optimize_configs.csv")

    # save results in df
    results = []

    # decide how many parameter combinations to search
    # for curr_config in configs.iloc[: args.num_search].itertuples():
    for curr_config in configs.itertuples():
        curr_config = curr_config._asdict()
        del curr_config["Index"]
        curr_config["model_name"] = args.model
        curr_config["data_split"] = args.data_split

        if args.model in ["gnn", "gat"]:
            curr_config["initialization"] = args.initialization

        with TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/config.json"
            with open(filename, "w", encoding="utf8") as fout:
                json.dump(curr_config, fout)
            locator = PathLocator(filename, args.output_path)
            curr_config["config_id"] = locator.hashid

        ground_truth, logits, best_metric = optimize_train_loop(
            locator,
            args.num_epochs,
            args.target_type,
            args.graph_type,
            args.input_root_dir,
            torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu"),
        )

        # e = Evaluator("inputs/validation_evaluation_params.json")
        e = Evaluator("configs/eval/validation_evaluation_params.json")
        val_metrics = e.evaluate(logits, ground_truth)
        results.append({**curr_config, **val_metrics})

    results = pd.json_normalize(results).sort_values(by=["F1"], ascending=False)
    results.to_csv(f"{args.output_path}/param_search.csv", index=False)
    writer = SummaryWriter(log_dir=args.output_path)
    writer.add_image(
        "F1 exploration",
        contour(results.drop("AUC", axis="columns")),
        dataformats="HWC",
    )
    writer.flush()
    print(results)


if __name__ == "__main__":
    main()
