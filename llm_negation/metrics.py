import pandas as pd


def topk_accuracy(predictions: pd.DataFrame, k: int = 1) -> float:
    filtered = predictions[
        (predictions["ctx_polarity"] == "aff") & (predictions["tgt_polarity"] == "aff")
    ][["target", "tokens"]]

    k = 5
    hit = 0
    for i, row in filtered.iterrows():
        if row["target"] in row["tokens"][:k]:
            hit += 1

    hit / len(filtered)


def ettinger_sensitivity(predictions: pd.DataFrame, polarity="aff") -> float:
    # get prediction for only aff contexts or only neg contexts
    scores_array = predictions[predictions["ctx_polarity"] == polarity][
        "target_logprob"
    ].array
    XX = scores_array[::2]  # appropriate predictions (eg. neg context neg target)
    XY = scores_array[1::2]  # inappropriate predictions (eg. neg context aff target)
    return (XX > XY).sum() / len(XX)


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
