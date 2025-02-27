import os
import json
import torch
import argparse
import pandas as pd
from minicons import scorer
from torch.utils.data import DataLoader

from config import MODELS, SCORER_CONFIG, DATASETS
from llm_negation.data import prepare_dataset_neg
from llm_negation.metrics import (
    ettinger_sensitivity,
    shivagunde_sensitivity,
    topk_accuracy,
)
from llm_negation.prediction import mlm_distribution, next_word_distribution


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--data", nargs="+", default=["data/NEG-136-SIMP.tsv"])
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to context (after WordNet prefix)",
    )
    parser.add_argument(
        "--wordnet_prefix",
        type=str,
        default="",
        help="WordNet prefix to add to context. Adds WN definition of aff and neg targets",
    )
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="conditional",
        choices=["conditional", "distribution"],
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--experiment_dir", type=str, default="experiment_1")
    parser.add_argument("--prediction_dir", type=str, default="predictions")
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DATA = args.data
    PREFIX = args.prefix
    WORDNET_PREFIX = args.wordnet_prefix
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    SKIP_IF_EXISTS = args.skip_if_exists
    PREDICTION_DIR = os.path.join(args.experiment_dir, args.prediction_dir)
    RESULTS_DIR = os.path.join(args.experiment_dir, args.results_dir)

    metrics = {}
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    metrics_temp_path = os.path.join(RESULTS_DIR, "metrics_temp.json")

    for model_type in MODELS.keys():
        for model in MODELS[model_type]:
            model_id = model.replace("/", "_")
            metrics[model_id] = {}

            for dataset_name in DATA:
                dataset_id = dataset_name.split("/")[-1].replace(".tsv", "")
                metrics[model_id][dataset_id] = {}

                prediction_dir = os.path.join(PREDICTION_DIR, dataset_id)
                if not os.path.exists(prediction_dir):
                    os.makedirs(prediction_dir)

                if SKIP_IF_EXISTS and os.path.exists(
                    os.path.join(prediction_dir, f"{model_id}.tsv")
                ):
                    continue

                print(f"Running experiment for {model_id} model")
                dataset = pd.read_csv(dataset_name, sep="\t")
                dataset = prepare_dataset_neg(
                    dataset, wordnet_prefix_word=WORDNET_PREFIX, prefix=PREFIX
                )

                model_type = next(
                    (type_ for type_, models in MODELS.items() if model in models), None
                )
                if model_type is None:
                    raise ValueError(f"Model {model} not found in config.MODELS.")

                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

                config = SCORER_CONFIG[model_type]

                scorer_args = {"device": DEVICE, **config["extra_args"]}
                scorer_ = config["scorer_class"](model, **scorer_args)

                if args.scoring_method == "distribution":
                    predictions = config["distribution_func"](dataloader, scorer_)

                elif args.scoring_method == "conditional":
                    predictions = config["conditional_func"](dataloader, scorer_)
                else:
                    raise ValueError(
                        f"Scoring method {args.scoring_method} not found in SCORER_CONFIG."
                    )

                del scorer_
                with torch.no_grad():
                    torch.cuda.empty_cache()

                if args.scoring_method == "distribution":
                    metrics[model_id][dataset_id]["top1"] = topk_accuracy(
                        predictions, k=1
                    )
                    metrics[model_id][dataset_id]["top3"] = topk_accuracy(
                        predictions, k=3
                    )
                    metrics[model_id][dataset_id]["top5"] = topk_accuracy(
                        predictions, k=5
                    )
                    metrics[model_id][dataset_id]["shivagunde"] = (
                        shivagunde_sensitivity(predictions)
                    )

                metrics[model_id][dataset_id]["ettinger_aff"] = ettinger_sensitivity(
                    predictions, polarity="aff"
                )
                metrics[model_id][dataset_id]["ettinger_neg"] = ettinger_sensitivity(
                    predictions, polarity="neg"
                )

                print(f"Model: {model_id}")
                print(f"Dataset: {dataset_id}")
                if args.scoring_method == "distribution":
                    print(f"Top-1 accuracy: {metrics[model_id][dataset_id]['top1']}")
                    print(f"Top-3 accuracy: {metrics[model_id][dataset_id]['top3']}")
                    print(f"Top-5 accuracy: {metrics[model_id][dataset_id]['top5']}")
                    print(
                        f"Shivagunde sensitivity: {metrics[model_id][dataset_id]['shivagunde']}"
                    )

                print(
                    f"Ettinger sensitivity (aff): {metrics[model_id][dataset_id]['ettinger_aff']}"
                )
                print(
                    f"Ettinger sensitivity (neg): {metrics[model_id][dataset_id]['ettinger_neg']}"
                )
                print("\n")

                if args.scoring_method == "distribution":
                    topk_predictions = predictions[["tokens", "logprobs"]]
                    predictions = predictions.drop(columns=["logprobs"])
                    predictions["tokens"] = predictions["tokens"].apply(lambda x: x[:5])

                    topk_predictions.to_csv(
                        os.path.join(prediction_dir, f"{model_id}_topk.tsv"),
                        sep="\t",
                        index=False,
                    )

                predictions.to_csv(
                    os.path.join(prediction_dir, f"{model_id}.tsv"),
                    sep="\t",
                    index=False,
                )

                if not os.path.exists(RESULTS_DIR):
                    os.makedirs(RESULTS_DIR)

                if os.path.exists(metrics_temp_path):
                    with open(metrics_temp_path, "r") as f:
                        existing_metrics = json.load(f)
                    existing_metrics.update(metrics)
                    with open(metrics_temp_path, "w") as f:
                        json.dump(existing_metrics, f, indent=4)
                else:
                    with open(metrics_temp_path, "w") as f:
                        json.dump(metrics, f, indent=4)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    if os.path.exists(metrics_temp_path):
        os.remove(metrics_temp_path)
