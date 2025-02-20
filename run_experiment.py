import os
import json
import torch
import pandas as pd
from minicons import scorer
from torch.utils.data import DataLoader

from config import MODELS, DATASETS
from llm_negation.data import prepare_dataset_neg
from llm_negation.metrics import (
    ettinger_sensitivity,
    shivagunde_sensitivity,
    topk_accuracy,
)
from llm_negation.prediction import mlm_distribution, next_word_distribution

DATA = [
    "data/NEG-136-SIMP.tsv",
    # "data/NEG-1500-SIMP-GEN.tsv",
    # "data/NEG-1500-SIMP-TEMP.tsv",
]
PREFIX = ""
WORDNET_PREFIX = ""
BATCH_SIZE = 4
DEVICE = "cuda"
SKIP_IF_EXISTS = False
PREDICTION_DIR = f"predictions/{WORDNET_PREFIX}"
RESULTS_DIR = f"results/{WORDNET_PREFIX}"

if __name__ == "__main__":
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

                if model in MODELS["CLM"]:
                    scorer_ = scorer.IncrementalLMScorer(model, DEVICE)
                    dataloader = DataLoader(
                        dataset, batch_size=BATCH_SIZE, shuffle=False
                    )
                    predictions = next_word_distribution(dataloader, scorer_)
                elif model in MODELS["MLM"]:
                    scorer_ = scorer.MaskedLMScorer(model, DEVICE)
                    dataloader = DataLoader(
                        dataset, batch_size=BATCH_SIZE, shuffle=False
                    )
                    predictions = mlm_distribution(dataloader, scorer_)
                elif model in MODELS["MAMBA"]:
                    scorer_ = scorer.MambaScorer(model, DEVICE)
                    dataloader = DataLoader(
                        dataset, batch_size=BATCH_SIZE, shuffle=False
                    )
                    predictions = next_word_distribution(dataloader, scorer_)
                elif model in MODELS["SEQ2SEQ"]:
                    scorer_ = scorer.Seq2SeqScorer(model, DEVICE)
                    dataloader = DataLoader(
                        dataset, batch_size=BATCH_SIZE, shuffle=False
                    )
                    predictions = next_word_distribution(dataloader, scorer_)
                else:
                    raise ValueError(f"Model {model} not found in config.MODELS.")

                del scorer_
                with torch.no_grad():
                    torch.cuda.empty_cache()

                metrics[model_id][dataset_id]["top1"] = topk_accuracy(predictions, k=1)
                metrics[model_id][dataset_id]["top3"] = topk_accuracy(predictions, k=3)
                metrics[model_id][dataset_id]["top5"] = topk_accuracy(predictions, k=5)
                metrics[model_id][dataset_id]["ettinger_aff"] = ettinger_sensitivity(
                    predictions, polarity="aff"
                )
                metrics[model_id][dataset_id]["ettinger_neg"] = ettinger_sensitivity(
                    predictions, polarity="neg"
                )
                metrics[model_id][dataset_id]["shivagunde"] = shivagunde_sensitivity(
                    predictions
                )

                print(f"Model: {model_id}")
                print(f"Dataset: {dataset_id}")
                print(f"Top-1 accuracy: {metrics[model_id][dataset_id]['top1']}")
                print(f"Top-3 accuracy: {metrics[model_id][dataset_id]['top3']}")
                print(f"Top-5 accuracy: {metrics[model_id][dataset_id]['top5']}")
                print(
                    f"Ettinger sensitivity (aff): {metrics[model_id][dataset_id]['ettinger_aff']}"
                )
                print(
                    f"Ettinger sensitivity (neg): {metrics[model_id][dataset_id]['ettinger_neg']}"
                )
                print(
                    f"Shivagunde sensitivity: {metrics[model_id][dataset_id]['shivagunde']}"
                )
                print("\n")

                topk_predictions = predictions[["tokens", "logprobs"]]
                predictions = predictions.drop(columns=["logprobs"])
                predictions["tokens"] = predictions["tokens"].apply(lambda x: x[:5])

                predictions.to_csv(
                    os.path.join(prediction_dir, f"{model_id}.tsv"),
                    sep="\t",
                    index=False,
                )
                topk_predictions.to_csv(
                    os.path.join(prediction_dir, f"{model_id}_topk.tsv"),
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
