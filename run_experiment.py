import os
import json
import yaml
import torch
import argparse
import pandas as pd
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader

from config import MODELS, SCORER_CONFIG
from llm_negation.data import prepare_dataset_neg
from llm_negation.metrics import calculate_metrics


@dataclass
class ExperimentConfig:
    dataset_paths: list[str]
    model_types: list[str]
    wordnet_prefix: str = ""
    prefix: str = ""
    suffix: str = ""
    batch_size: int = 4
    topk: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_dir: str = "experiments"
    prediction_dir: str = "predictions"
    skip_if_exists: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(
            dataset_paths=args.data,
            model_types=args.model_type,
            wordnet_prefix=args.wordnet_prefix,
            prefix=args.prefix,
            suffix=args.suffix,
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
    parser.add_argument(
        "--save_config",
        type=str,
        default="experiment_config.yaml",
        help="Save parsed arguments to YAML config file",
    )
    parser.add_argument("--data", nargs="+", required=True, help="Dataset path(s)")
    parser.add_argument("--model_type", nargs="+", required=True, help="Model type(s)")
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix to add to context"
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix to add to context"
    )
    parser.add_argument(
        "--wordnet_prefix", type=str, default="", help="WordNet prefix type"
    )
    parser.add_argument("--scoring_method", type=str, default="distribution")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--topk", type=int, default=30, help="Number of top predictions to keep"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--skip_if_exists", action="store_true", help="Skip if results exist"
    )
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--prediction_dir", type=str, default="predictions")

    return parser.parse_args()


def run_experiment(config: ExperimentConfig):
    args = parse_args()

    DATA = args.data
    MODEL_TYPES = args.model_type
    PREFIX = args.prefix
    SUFFIX = args.suffix
    WORDNET_PREFIX = args.wordnet_prefix
    SCORING_METHOD = args.scoring_method
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    SKIP_IF_EXISTS = args.skip_if_exists
    EXPERIMENT_DIR = args.experiment_dir
    PREDICTION_DIR = os.path.join(EXPERIMENT_DIR, args.prediction_dir)
    SAVE_CONFIG_PATH = os.path.join(EXPERIMENT_DIR, args.save_config)

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    config.to_yaml(SAVE_CONFIG_PATH)

    metrics = {}
    metrics_path = os.path.join(EXPERIMENT_DIR, "metrics.json")

    for model_type in MODEL_TYPES:
        for model in MODELS[model_type]:
            model_id = model.replace("/", "_")
            metrics[model_id] = {}

            for dataset_name in DATA:
                ##################################
                ########## LOAD DATASET ##########
                ##################################
                dataset_id = dataset_name.split("/")[-1].replace(".tsv", "")
                metrics[model_id][dataset_id] = {}

                dataset_prediction_dir = os.path.join(PREDICTION_DIR, dataset_id)
                predictions_path = os.path.join(
                    dataset_prediction_dir, f"{model_id}.tsv"
                )
                topk_predictions_path = os.path.join(
                    dataset_prediction_dir, f"{model_id}_topk.tsv"
                )

                os.makedirs(dataset_prediction_dir, exist_ok=True)
                if SKIP_IF_EXISTS and os.path.exists(
                    os.path.join(dataset_prediction_dir, f"{model_id}.tsv")
                ):
                    continue

                print(f"Running experiment for {model_id} model")
                dataset = pd.read_csv(dataset_name, sep="\t")
                dataset = prepare_dataset_neg(
                    dataset,
                    wordnet_prefix_word=WORDNET_PREFIX,
                    prefix=PREFIX,
                    suffix=SUFFIX,
                )

                ######################################
                ########## MAKE PREDICTIONS ##########
                ######################################
                model_type = next(
                    (type_ for type_, models in MODELS.items() if model in models), None
                )
                if model_type is None:
                    raise ValueError(f"Model {model} not found in config.MODELS.")

                scorer_config = SCORER_CONFIG[model_type]
                scorer_args = {"device": DEVICE, **scorer_config["extra_args"]}
                scorer_ = scorer_config["scorer_class"](model, **scorer_args)

                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
                if SCORING_METHOD == "distribution":
                    predictions = scorer_config["distribution_func"](
                        dataloader, scorer_
                    )

                elif SCORING_METHOD == "conditional":
                    predictions = scorer_config["conditional_func"](dataloader, scorer_)
                else:
                    raise ValueError(
                        f"Scoring method {SCORING_METHOD} not found in SCORER_CONFIG."
                    )

                del scorer_
                with torch.no_grad():
                    torch.cuda.empty_cache()

                #######################################
                ########## CALCULATE METRICS ##########
                #######################################
                print(f"Model: {model_id}")
                print(f"Dataset: {dataset_id}")
                metrics = calculate_metrics(
                    predictions, metrics, model_id, dataset_id, show=True
                )

                if args.scoring_method == "distribution":
                    topk_predictions = predictions[["tokens", "logprobs"]]
                    predictions = predictions.drop(columns=["logprobs"])
                    predictions["tokens"] = predictions["tokens"].apply(lambda x: x[:5])

                    topk_predictions.to_csv(
                        topk_predictions_path, sep="\t", index=False
                    )

                predictions.to_csv(predictions_path, sep="\t", index=False)

                ##################################
                ########## SAVE METRICS ##########
                ##################################
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        existing_metrics = json.load(f)
                        existing_metrics.update(metrics)
                    with open(metrics_path, "w") as f:
                        json.dump(existing_metrics, f, indent=4)
                else:
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=4)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def main():
    args = parse_args()
    config = ExperimentConfig.from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()
