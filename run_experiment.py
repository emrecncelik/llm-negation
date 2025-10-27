import os
import json
import yaml
import torch
import argparse
import pandas as pd
from time import time
from datetime import timedelta
from minicons import scorer
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from config import MODELS, get_model_by_type
from llm_negation.data import prepare_negation_dataset
from llm_negation.metrics import calculate_metrics

@dataclass
class ExperimentConfig:
    config: str
    data_path: list[str]
    model_type: list[str]
    prompt_template: str = "{context} {determiner}"
    assistant_message_template: str = None
    scoring_method: str = "distribution"
    batch_size: int = 4
    topk: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_dir: str = "experiments"
    prediction_dir: str = "predictions"
    skip_if_exists: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(
            config=args.config,
            data_path=args.data_path,
            model_type=args.model_type,
            prompt_template=args.prompt_template,
            assistant_message_template=args.assistant_message_template,
            scoring_method=args.scoring_method,
            batch_size=args.batch_size,
            topk=args.topk,
            device=args.device,
            experiment_dir=args.experiment_dir,
            prediction_dir=args.prediction_dir,
            skip_if_exists=args.skip_if_exists,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        try:
            with open(yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)

        except yaml.YAMLError:
            raise ValueError(f"Invalid YAML format in config file: {yaml_path}")

    def to_yaml(self, yaml_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
        config_dict = asdict(self)
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        except (IOError, OSError) as e:
            raise IOError(f"Error writing config to file: {e}")

        print(f"Configuration saved to {yaml_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run negation experiment")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--data_path", nargs="+", help="Dataset path(s)")
    parser.add_argument("--model_type", nargs="+", help="Model type(s)")
    parser.add_argument("--prompt_template", type=str, default="{context} {determiner}")
    parser.add_argument("--assistant_message_template", type=str, default=None)
    parser.add_argument("--scoring_method", type=str, default="distribution", choices=["distribution", "sequence_score"])
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--topk", type=int, default=30, help="Number of top predictions to keep")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if results exist")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--prediction_dir", type=str, default="predictions")

    return parser.parse_args()


def run_experiment(config: ExperimentConfig):
    config.prediction_dir = os.path.join(config.experiment_dir, config.prediction_dir)

    os.makedirs(config.experiment_dir, exist_ok=True)
    os.makedirs(config.prediction_dir, exist_ok=True)
    config.to_yaml(os.path.join(config.experiment_dir, "config.yaml"))

    metrics = {}
    metrics_path = os.path.join(config.experiment_dir, "metrics.json")

    for model_type in config.model_type:
        for model_config in get_model_by_type(model_type):
            metrics[model_config.id] = {}

            for dataset_name in config.data_path:
                ##################################
                ########## LOAD DATASET ##########
                ##################################
                dataset_id = dataset_name.split("/")[-1].replace(".tsv", "")

                # Setup directories and paths for predictions
                dataset_prediction_dir = os.path.join(config.prediction_dir, dataset_id)
                predictions_path = os.path.join(
                    dataset_prediction_dir, f"{model_config.id}.tsv"
                )
                topk_predictions_path = os.path.join(
                    dataset_prediction_dir, f"{model_config.id}_topk.tsv"
                )

                os.makedirs(dataset_prediction_dir, exist_ok=True)
                if config.skip_if_exists and os.path.exists(predictions_path):
                    print(
                        f"Skipping experiment for {model_config.id} model and {dataset_id} dataset"
                    )
                    continue

                print(f"Running experiment for {model_config.id} model")
                
                if model_type == "MAMBA":
                    tokenizer = AutoTokenizer.from_pretrained(model_config.scorer_args["tokenizer"])
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_config.ckpt)

                dataset = pd.read_csv(dataset_name, sep="\t")
                dataset = prepare_negation_dataset(
                    tokenizer=tokenizer,
                    dataset=dataset,
                    prompt_template=config.prompt_template,
                    assistant_message_template=config.assistant_message_template,
                )
                dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

                ######################################
                ########## MAKE PREDICTIONS ##########
                ######################################
                if model_type in ["ICLM", "CLM", "RANDOM"]:
                    scorer_class = scorer.IncrementalLMScorer
                elif model_type == "MAMBA":
                    scorer_class = scorer.MambaScorer
                elif model_type == "MLM":
                    scorer_class = scorer.MaskedLMScorer
                elif model_type == "SEQ2SEQ":
                    scorer_class = scorer.Seq2SeqScorer
                else:
                    raise ValueError(f"Model type {model_type} not found in config.MODELS.")

                scorer_args = {"device": config.device, **model_config.scorer_args}
                sc = scorer_class(model_config.ckpt, trust_remote_code=True, **scorer_args)

                if config.scoring_method == "distribution":
                    predictions = model_config.distribution_function(
                        dataloader=dataloader, scorer=sc, model_type=model_type, topk=config.topk
                    )

                elif config.scoring_method == "sequence_score":
                    predictions = model_config.sequence_score_function(
                        dataloader=dataloader, scorer=sc, model_type=model_type
                    )
                else:
                    raise ValueError(
                        f"Scoring method {config.scoring_method} not found."
                    )

                del sc
                with torch.no_grad():
                    torch.cuda.empty_cache()

                #######################################
                ########## CALCULATE METRICS ##########
                #######################################
                print(f"Model: {model_config.id}")
                print(f"Dataset: {dataset_id}")

                metrics[model_config.id][dataset_id] = {}
                metrics = calculate_metrics(
                    predictions, metrics, model_config.id, dataset_id, show=True
                )

                if config.scoring_method == "distribution":
                    topk_predictions = predictions[["tokens", "logprobs"]]
                    predictions = predictions.drop(columns=["logprobs"])
                    predictions["tokens"] = predictions["tokens"].apply(lambda x: x[:5])
                    topk_predictions.to_csv(topk_predictions_path, sep="\t", index=False)

                predictions.to_csv(predictions_path, sep="\t", index=False)

                ##################################
                ########## SAVE METRICS ##########
                ##################################
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        existing_metrics = json.load(f)
                    
                    combined_metrics = existing_metrics.copy()
                    
                    for model_id, model_metrics in metrics.items():
                        if model_id not in combined_metrics:
                            combined_metrics[model_id] = {}
                        
                        for dataset_id, dataset_metrics in model_metrics.items():
                            combined_metrics[model_id][dataset_id] = dataset_metrics
                    
                    with open(metrics_path, "w") as f:
                        json.dump(combined_metrics, f, indent=4)
                else:
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=4)


def main():
    args = parse_args()
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig.from_args(args)

    start_time = time()
    run_experiment(config)
    elapsed_time = time() - start_time
    time_delta = timedelta(seconds=elapsed_time)
    print(f"Experiment finished in {time_delta}")


if __name__ == "__main__":
    main()
