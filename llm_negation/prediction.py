import pandas as pd
from tqdm import tqdm
from typing import Union
from minicons import scorer
from torch.utils.data import DataLoader


def apply_chat_template(contexts: list[str], scorer: scorer.LMScorer, model_type: str):
    if scorer.tokenizer.chat_template is not None and model_type == "ICLM":
        messages = [[{"role": "user", "content": c}] for c in contexts]
        contexts = scorer.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return contexts


def cleanup_tokens(
    sc: scorer.LMScorer,
    tokens: list[list[str]],
) -> list[list[str]]:
    if not isinstance(sc, scorer.MambaScorer):
        model_name = sc.model.config._name_or_path.lower()
    else:
        model_name = "mamba"
    if any([m in model_name for m in ("gemma", "albert", "t5")]):
        tokens = [list(map(lambda x: x.replace("▁", "").strip(), t)) for t in tokens]
    if any(
        [
            m in model_name
            for m in (
                "modernbert",
                "roberta",
                "gpt2",
                "llama",
                "pythia",
                "mamba",
                "qwen",
            )
        ]
    ):
        tokens = [list(map(lambda x: x.replace("Ġ", "").strip(), t)) for t in tokens]
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


def conditional_score(
    dataloader: DataLoader,
    scorer: scorer.LMScorer,
    model_type: str,
    reduction: callable = lambda x: x.mean(0).item(),  # default from minicons
):
    def batch_preprocess(batch):
        contexts = list(batch[0])
        targets = [t + "." for t in batch[1]]  # end of sentence
        ctx_polarity = batch[2]
        tgt_polarity = batch[3]

        contexts = apply_chat_template(contexts, scorer, model_type)
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
        # inputs = [f"{c} {t}." for c, t in zip(contexts, targets)]
        outputs = scorer.conditional_score(
            prefix=contexts, stimuli=targets, reduction=reduction
        )
        # outputs = scorer.sequence_score(inputs, reduction=reduction)
        predictions["context"].extend(contexts)
        predictions["target"].extend(targets)
        predictions["target_logprob"].extend(outputs)
        predictions["ctx_polarity"].extend(ctx_polarity)
        predictions["tgt_polarity"].extend(tgt_polarity)

    return pd.DataFrame(predictions)


def next_word_distribution(
    dataloader: DataLoader,
    scorer: Union[scorer.IncrementalLMScorer, scorer.MambaScorer, scorer.Seq2SeqScorer],
    model_type: str,
    topk: int = 300,
) -> pd.DataFrame:
    def batch_preprocess(batch):
        contexts = batch[0]
        targets = batch[1]
        ctx_polarity = batch[2]
        tgt_polarity = batch[3]

        contexts = apply_chat_template(contexts, scorer, model_type)
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
        outputs = scorer.next_word_distribution(contexts)
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
            scorer, tokens
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
    placeholder: str = "token",  # this is dumb
) -> pd.DataFrame:
    if model_type not in ["SEQ2SEQ", "MLM"]:
        raise ValueError(f"Model type {model_type} not supported for mlm_distribution.")

    def batch_preprocess(batch):
        contexts = batch[0]
        targets = batch[1]
        ctx_polarity = batch[2]
        tgt_polarity = batch[3]
        # if target occurs in the context more than two times
        # scorer.cloze_distribution will fail
        contexts = [
            f"{c} {placeholder}." for c in contexts
        ]  # also adding period (signaling this is the last token of the sent. for MLM)
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
        inputs = [(c, placeholder) for c in contexts]
        outputs = scorer.cloze_distribution(inputs)
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
            scorer, tokens
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
