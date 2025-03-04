import pandas as pd
from tqdm import tqdm
from typing import Union
from minicons import scorer
from torch.utils.data import DataLoader


def cleanup_tokens(
    tokens: list[list[str]],
) -> list[list[str]]:
    tokens = [
        list(map(lambda x: x.replace("Ġ", "").replace("▁", "").strip(), t))
        for t in tokens
    ]
    return tokens


def get_variation_logprobs(target, tokenizer, logprobs):
    idx = 0
    variations = [
        target,
        f" {target}",
        f"{target.capitalize()}",
        f" {target.capitalize()}",
    ]
    ids = [
        id_[idx] for id_ in tokenizer(variations, add_special_tokens=False).input_ids
    ]
    variation_logprobs = logprobs[ids]
    best_variation_logprob, best_variation_id = (
        variation_logprobs.max(),
        variation_logprobs.argmax(),
    )
    return best_variation_logprob, tokenizer.convert_ids_to_tokens(
        ids[best_variation_id]
    )


def sequence_score(
    dataloader: DataLoader,
    scorer: scorer.LMScorer,
    model_type: str,
    reduction: callable = lambda x: x.mean(0).item(),  # default from minicons
):
    if model_type not in ["ICLM", "CLM", "MAMBA", "MLM"]:
        raise ValueError(f"Model type {model_type} not supported for sequence_score.")

    def batch_preprocess(batch):
        contexts = batch[0]
        targets = batch[1]
        ctx_polarity = batch[2]
        tgt_polarity = batch[3]

        return contexts, targets, ctx_polarity, tgt_polarity

    predictions = {
        "context": [],
        "target": [],
        "target_logprob": [],  # this is actually conditional & reduced logprobs
        "ctx_polarity": [],
        "tgt_polarity": [],
    }

    for batch in tqdm(dataloader):
        contexts, targets, ctx_polarity, tgt_polarity = batch_preprocess(batch)
        outputs = scorer.sequence_score(list(contexts), reduction=reduction)
        predictions["context"].extend(contexts)
        predictions["target"].extend(targets)
        predictions["target_logprob"].extend(outputs)
        predictions["ctx_polarity"].extend(ctx_polarity)
        predictions["tgt_polarity"].extend(tgt_polarity)

    return pd.DataFrame(predictions)


def next_word_distribution(
    dataloader: DataLoader,
    scorer: Union[scorer.IncrementalLMScorer, scorer.MambaScorer],
    model_type: str,
    topk: int = 300,
) -> pd.DataFrame:
    if model_type not in ["ICLM", "CLM", "MAMBA"]:
        raise ValueError(
            f"Model type {model_type} not supported for next_word_distribution."
        )

    def batch_preprocess(batch):
        contexts = batch[0]
        targets = batch[1]
        ctx_polarity = batch[2]
        tgt_polarity = batch[3]
        return contexts, targets, ctx_polarity, tgt_polarity

    predictions = {
        "context": [],
        "target": [],
        "target_logprob": [],
        "best_target_variation": [],
        "logprobs": [],
        "tokens": [],
        "ctx_polarity": [],
        "tgt_polarity": [],
    }
    for batch in tqdm(dataloader):
        contexts, targets, ctx_polarity, tgt_polarity = batch_preprocess(batch)
        outputs = scorer.next_word_distribution(list(contexts))
        topk_preds = outputs.topk(topk)

        target_logprobs = []
        variation_used = []
        for target, logprobs in zip(targets, outputs.detach().cpu().numpy()):
            best_variation_logprob, best_variation = get_variation_logprobs(
                target, scorer.tokenizer, logprobs
            )
            target_logprobs.append(best_variation_logprob)
            variation_used.append(best_variation)

        tokens = topk_preds.indices.detach().cpu().numpy()
        logprobs = topk_preds.values.detach().cpu().numpy()
        tokens = [scorer.tokenizer.convert_ids_to_tokens(t) for t in tokens]
        tokens = cleanup_tokens(
            tokens
        )  # some models use special chars to denote preceding spaces, remove them

        predictions["context"].extend(contexts)
        predictions["target"].extend(targets)
        predictions["target_logprob"].extend(target_logprobs)
        predictions["best_target_variation"].extend(variation_used)
        predictions["tokens"].extend(tokens)
        predictions["logprobs"].extend(logprobs)
        predictions["ctx_polarity"].extend(ctx_polarity)
        predictions["tgt_polarity"].extend(tgt_polarity)

    predictions = pd.DataFrame(predictions)
    return predictions


def mlm_distribution(
    dataloader: DataLoader,
    scorer: Union[scorer.MaskedLMScorer, scorer.Seq2SeqScorer],
    model_type: str,
    topk: int = 300,
) -> pd.DataFrame:
    if model_type not in ["SEQ2SEQ", "MLM"]:
        raise ValueError(f"Model type {model_type} not supported for mlm_distribution.")

    def batch_preprocess(batch):
        contexts = [" ".join(c.split()[:-1]) + " token." for c in batch[0]]
        targets = batch[1]
        ctx_polarity = batch[2]
        tgt_polarity = batch[3]
        return contexts, targets, ctx_polarity, tgt_polarity

    predictions = {
        "context": [],
        "target": [],
        "target_logprob": [],
        "best_target_variation": [],
        "logprobs": [],
        "tokens": [],
        "ctx_polarity": [],
        "tgt_polarity": [],
    }
    for batch in tqdm(dataloader):
        contexts, targets, ctx_polarity, tgt_polarity = batch_preprocess(batch)
        outputs = scorer.cloze_distribution(
            list(zip(contexts, ["token"] * len(contexts)))
        )
        topk_preds = outputs.topk(topk)

        target_logprobs = []
        variation_used = []
        for target, logprobs in zip(targets, outputs.detach().cpu().numpy()):
            best_variation_logprob, best_variation = get_variation_logprobs(
                target, scorer.tokenizer, logprobs
            )
            target_logprobs.append(best_variation_logprob)
            variation_used.append(best_variation)

        tokens = topk_preds.indices.detach().cpu().numpy()
        logprobs = topk_preds.values.detach().cpu().numpy()
        tokens = [scorer.tokenizer.convert_ids_to_tokens(t) for t in tokens]
        tokens = cleanup_tokens(
            tokens
        )  # some models use special chars to denote preceding spaces, remove them

        predictions["context"].extend(contexts)
        predictions["target"].extend(targets)
        predictions["target_logprob"].extend(target_logprobs)
        predictions["best_target_variation"].extend(variation_used)
        predictions["tokens"].extend(tokens)
        predictions["logprobs"].extend(logprobs)
        predictions["ctx_polarity"].extend(ctx_polarity)
        predictions["tgt_polarity"].extend(tgt_polarity)

    predictions = pd.DataFrame(predictions)
    return predictions
