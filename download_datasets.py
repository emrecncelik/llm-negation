import os
import requests
import pandas as pd
from config import DATASETS
import argparse

parser = argparse.ArgumentParser(description="Download and format datasets.")
parser.add_argument(
    "--data_dir", type=str, default="data", help="Directory to save the datasets."
)
args = parser.parse_args()


def download_file(url, filename, data_dir):
    response = requests.get(url)
    response.raise_for_status()

    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, filename)

    with open(filename, "wb") as file:
        file.write(response.content)
    print(f"Downloaded {filename}")


def format_negation(file_dir: str) -> None:
    """
    Formats the negation dataset to original data format
    by splitting the context into affirmative and negative parts,
    replacing the second-to-last word with "(a|an)", and extracting
    the last word as the target.

    Args:
        file_dir (str): The file directory of the dataset.

    Returns:
        None
    """
    dataset = pd.read_csv(file_dir, header=None)
    positive = dataset[0][[i for i in range(0, 1500, 2)]]
    negative = dataset[0][[i for i in range(1, 1500, 2)]]

    dataset["context_aff"] = positive.reset_index(drop=True)
    dataset["context_neg"] = negative.reset_index(drop=True)
    dataset = dataset.drop(columns=[0, 1])
    dataset = dataset.dropna()
    dataset["target_aff"] = ""
    dataset["target_neg"] = ""

    for i, row in dataset.iterrows():
        aff_split = row["context_aff"].split()
        neg_split = row["context_neg"].split()
        aff_last_word = aff_split[-1]
        neg_last_word = neg_split[-1]
        aff_split[-2] = "(a|an)"
        neg_split[-2] = "(a|an)"
        context_aff = " ".join(aff_split[:-1])
        context_neg = " ".join(neg_split[:-1])

        dataset.loc[i, "context_aff"] = context_aff
        dataset.loc[i, "context_neg"] = context_neg
        dataset.loc[i, "target_aff"] = aff_last_word
        dataset.loc[i, "target_neg"] = neg_last_word

    dataset.to_csv(file_dir.replace("txt", "tsv"), sep="\t", index=False)


if __name__ == "__main__":
    args = parser.parse_args()

    for dataset in DATASETS.values():
        download_file(dataset["url"], dataset["filename"], args.data_dir)
        if dataset["format"]:
            format_negation(os.path.join(args.data_dir, dataset["filename"]))

    negnat = pd.read_csv(os.path.join(args.data_dir, "NEG-136-NAT.tsv"), sep="\t")
    negnat[negnat["licensing"] == "Y"].to_csv(
        os.path.join(args.data_dir, "NEG-136-NAT-NT.tsv"), sep="\t", index=False
    )
    negnat[negnat["licensing"] == "N"].to_csv(
        os.path.join(args.data_dir, "NEG-136-NAT-LN.tsv"), sep="\t", index=False
    )
