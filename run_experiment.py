import os
import json
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
BATCH_SIZE = 8
DEVICE = "cuda"


if __name__ == "__main__":
    metrics = {}
    for model in MODELS["MLM"]:
        model_id = model.replace("/", "_")
        metrics[model_id] = {}
        for dataset_name in DATA:
            dataset_id = dataset_name.split("/")[-1].replace(".tsv", "")
            metrics[model_id][dataset_id] = {}
            print(f"Running experiment for {model} model")
            dataset = pd.read_csv(dataset_name, sep="\t")
            dataset = prepare_dataset_neg(dataset, wordnet_prefix=False)

            if model in MODELS["CLM"]:
                scorer_ = scorer.IncrementalLMScorer(model, DEVICE)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
                predictions = next_word_distribution(dataloader, scorer_)
            elif model in MODELS["MLM"]:
                scorer_ = scorer.MaskedLMScorer(model, DEVICE)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
                predictions = mlm_distribution(dataloader, scorer_)
            elif model in MODELS["MAMBA"]:
                scorer_ = scorer.MambaScorer(model, DEVICE)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
                predictions = mlm_distribution(dataloader, scorer_)
            elif model in MODELS["SEQ2SEQ"]:
                scorer_ = scorer.Seq2SeqScorer(model, DEVICE)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
                predictions = mlm_distribution(dataloader, scorer_)
            else:
                raise ValueError(f"Model {model_id} not found in config.MODELS.")

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
            print(f"Top-1 accuracy: {metrics[model][dataset_id]['top1']}")
            print(f"Top-3 accuracy: {metrics[model][dataset_id]['top3']}")
            print(f"Top-5 accuracy: {metrics[model][dataset_id]['top5']}")
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

            prediction_dir = os.path.join("predictions", dataset_id)
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir)

            predictions.to_csv(
                os.path.join(prediction_dir, f"{model_id}.tsv"),
                sep="\t",
                index=False,
            )
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
