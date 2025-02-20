import pandas as pd


def topk_accuracy(predictions: pd.DataFrame, k: int = 1) -> float:
    filtered = predictions[
        (predictions["ctx_polarity"] == "aff") & (predictions["tgt_polarity"] == "aff")
    ][["target", "tokens"]]
    hit = 0
    for i, row in filtered.iterrows():
        if row["target"] in row["tokens"][:k]:
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
