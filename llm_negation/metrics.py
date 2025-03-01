import pandas as pd


def calculate_metrics(
    predictions: pd.DataFrame,
    metrics: dict,
    model_id: str,
    dataset_id: str,
    show: bool = False,
) -> dict:
    if "tokens" in predictions.columns:
        metrics[model_id][dataset_id]["top1"] = topk_accuracy(predictions, k=1)
        metrics[model_id][dataset_id]["top3"] = topk_accuracy(predictions, k=3)
        metrics[model_id][dataset_id]["top5"] = topk_accuracy(predictions, k=5)
        metrics[model_id][dataset_id]["shivagunde"] = shivagunde_sensitivity(
            predictions
        )

    metrics[model_id][dataset_id]["ettinger_aff"] = ettinger_sensitivity(
        predictions, polarity="aff"
    )
    metrics[model_id][dataset_id]["ettinger_neg"] = ettinger_sensitivity(
        predictions, polarity="neg"
    )

    if show:
        if "tokens" in predictions.columns:
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

    return metrics


def topk_accuracy(predictions: pd.DataFrame, k: int = 1) -> float:
    filtered = predictions[
        (predictions["ctx_polarity"] == "aff") & (predictions["tgt_polarity"] == "aff")
    ][["target", "tokens"]]
    hit = 0

    for i, row in filtered.iterrows():
        if row["target"].lower() in list(map(lambda x: x.lower(), row["tokens"][:k])):
            hit += 1

    return hit / len(filtered)


def ettinger_sensitivity(predictions: pd.DataFrame, polarity="aff") -> float:
    if polarity == "aff":
        other = "neg"
    elif polarity == "neg":
        other = "aff"

    # get prediction for only aff contexts or only neg contexts
    XX_scores = predictions[
        (predictions["ctx_polarity"] == polarity)
        & (predictions["tgt_polarity"] == polarity)
    ]["target_logprob"].array

    XY_scores = predictions[
        (predictions["ctx_polarity"] == polarity)
        & (predictions["tgt_polarity"] == other)
    ]["target_logprob"].array

    return (XX_scores > XY_scores).sum() / len(XX_scores)


def shivagunde_sensitivity(predictions: pd.DataFrame) -> float:
    # both affirmative tgt for a/an match,
    # if a/an does not match it's not sensitivity to negation but to determiner
    neg_tokens = (
        predictions[
            (predictions["ctx_polarity"] == "neg")
            & (predictions["tgt_polarity"] == "aff")
        ]["tokens"]
        .apply(lambda x: x[0])
        .array
    )
    aff_tokens = (
        predictions[
            (predictions["ctx_polarity"] == "aff")
            & (predictions["tgt_polarity"] == "aff")
        ]["tokens"]
        .apply(lambda x: x[0])
        .array
    )
    return (aff_tokens != neg_tokens).sum() / len(aff_tokens)
