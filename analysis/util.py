import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

def custom_float_format(x):
    if x.is_integer():
        return f'{int(x)}'
    else:
        return f'{x:.1f}'

def get_metric_for_datasets(
    df: pd.DataFrame,
    dataset_names: list[str] = [
        "NEG-136-SIMP",
        "NEG-1500-SIMP-TEMP",
        "NEG-1500-SIMP-GEN",
    ],
    metric: str = "top1",
):
    metrics = []
    for dataset_name in dataset_names:
        metrics.append(df[(dataset_name, metric)])
    return tuple(metrics)


def reorder_dataframe_by_model_list(df: pd.DataFrame, ordered_models: list[str]):
    model_order = {model: i for i, model in enumerate(ordered_models)}
    df = df.copy()
    df["_order"] = df["Model"].map(model_order)

    sorted_df = df.sort_values("_order")
    sorted_df = sorted_df.drop("_order", axis=1)
    return sorted_df


def organize_results(path: str):
    df = pd.read_json(path)
    df = df.T
    df_normalized = pd.concat(
        [df[col].apply(pd.Series).add_prefix(col + "%") for col in df.columns], axis=1
    )
    df_normalized.columns = pd.MultiIndex.from_tuples(
        [(col.split("%")[0], col.split("%")[1]) for col in df_normalized.columns]
    )
    df_normalized = (
        df_normalized.reset_index()
        .rename(columns={"index": "Model"})
        .sort_values(by="Model")
        .reset_index(drop=True)
    )
    return df_normalized


def replace_cloze_with_sequence(
    sequence_df: pd.DataFrame,
    cloze_df: pd.DataFrame,
    dataset_names: list[str] = [
        "NEG-136-SIMP",
        "NEG-1500-SIMP-TEMP",
        "NEG-1500-SIMP-GEN",
    ],
    metric_names: list[str] = ["ettinger_aff", "ettinger_neg"],
):
    cloze_df = cloze_df.copy()
    for dataset_name in dataset_names:
        for metric_name in metric_names:
            cloze_df[(dataset_name, metric_name)] = sequence_df[
                (dataset_name, metric_name)
            ]
    return cloze_df


def add_color_to_table(latex_table_path: str):
    with open(latex_table_path, "r") as file:
        latex_table = file.read()

    tabular_start = latex_table.find("\\begin{tabular}")
    tabular_end = latex_table.find("\\end{tabular}")

    if tabular_start == -1 or tabular_end == -1:
        return "Could not find tabular environment"

    before_table = latex_table[:tabular_start]
    table_content = latex_table[tabular_start : tabular_end + len("\\end{tabular}")]
    after_table = latex_table[tabular_end + len("\\end{tabular}") :]

    header_end = table_content.find("\\midrule")
    if header_end == -1:
        return "Could not find header separator (\\midrule)"

    header = table_content[: header_end + len("\\midrule")]
    body = table_content[header_end + len("\\midrule") :]

    pattern = r"(&\s*)(\d+(?:\.\d+)?)(\s*(?=&|\\\\))"

    def replace_with_color(match):
        prefix = match.group(1)
        number = match.group(2)
        suffix = match.group(3)
        return f"{prefix}\\colorByValue{{{number}}}{suffix}"

    colored_body = re.sub(pattern, replace_with_color, body)
    colored_table = before_table + header + colored_body + after_table
    colored_table = colored_table.replace("\\textbackslash{}", "\\").replace("\{", "{").replace("\}", "}")

    with open(latex_table_path, "w") as file:
        file.write(colored_table)

    return colored_table


def extract_model_size_from_hf(ckpt, delay=0.5):
    time.sleep(delay)
    url = "https://huggingface.co/" + ckpt
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch page: {url} (status code: {response.status_code})"
        )

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n")

    pattern = re.compile(r"Model size\s*([\d\.]+[MB])\s*params", re.IGNORECASE)
    match = pattern.search(text)

    if match:
        return match.group(1)
    else:
        return None


def extract_model_size_from_name(id):
    pattern = re.compile(r"([\d\.]+[MB])", re.IGNORECASE)
    match = pattern.search(id)
    return match
