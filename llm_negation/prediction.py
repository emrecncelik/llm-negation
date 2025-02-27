import pandas as pd
from tqdm import tqdm
from typing import Union
from minicons import scorer
from torch.utils.data import DataLoader


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


def conditional_score(
    dataloader: DataLoader,
    scorer: scorer.LMScorer,
    reduction: callable = lambda x: x.mean(0).item(),  # default from minicons
):
    def batch_preprocess(batch):
        contexts = list(batch[0])
        targets = [t + "." for t in batch[1]]  # end of sentence
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
        outputs = scorer.conditional_score(
            prefix=contexts, stimuli=targets, reduction=reduction
        )

        predictions["context"].extend(contexts)
        predictions["target"].extend(targets)
        predictions["target_logprob"].extend(outputs)
        predictions["ctx_polarity"].extend(ctx_polarity)
        predictions["tgt_polarity"].extend(tgt_polarity)

    return pd.DataFrame(predictions)


def next_word_distribution(
    dataloader: DataLoader,
    scorer: Union[scorer.IncrementalLMScorer, scorer.MambaScorer, scorer.Seq2SeqScorer],
    topk: int = 300,
) -> pd.DataFrame:
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
        "logprobs": [],
        "tokens": [],
        "ctx_polarity": [],
        "tgt_polarity": [],
    }
    for batch in tqdm(dataloader):
        contexts, targets, ctx_polarity, tgt_polarity = batch_preprocess(batch)
        inputs = contexts
        outputs = scorer.next_word_distribution(inputs)

        # get first token of the targets,
        # does not matter if target is single token
        # targets = [
        #     scorer.tokenizer.decode(
        #         scorer.tokenizer(target, add_special_tokens=False)["input_ids"][0]
        #     )
        #     for target in targets
        # ]

        target_indices = [
            scorer.tokenizer(target, add_special_tokens=False)["input_ids"][0]
            for target in targets
        ]
        target_logprobs = [
            outputs[idx][i].detach().cpu().numpy()
            for idx, i in enumerate(target_indices)
        ]

        topk_preds = outputs.topk(topk)
        tokens = topk_preds.indices.detach().cpu().numpy()
        logprobs = topk_preds.values.detach().cpu().numpy()
        tokens = [scorer.tokenizer.convert_ids_to_tokens(t) for t in tokens]
        tokens = cleanup_tokens(
            scorer, tokens
        )  # some models use special chars to denote preceding spaces, remove them

        predictions["context"].extend(contexts)
        predictions["target"].extend(targets)
        predictions["target_logprob"].extend(target_logprobs)
        predictions["tokens"].extend(tokens)
        predictions["logprobs"].extend(logprobs)
        predictions["ctx_polarity"].extend(ctx_polarity)
        predictions["tgt_polarity"].extend(tgt_polarity)

    predictions = pd.DataFrame(predictions)
    return predictions


def mlm_distribution(
    dataloader: DataLoader,
    scorer: Union[scorer.MaskedLMScorer, scorer.Seq2SeqScorer],
    topk: int = 300,
    placeholder: str = "token",  # this is dumb
) -> pd.DataFrame:
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
        "logprobs": [],
        "tokens": [],
        "ctx_polarity": [],
        "tgt_polarity": [],
    }
    for batch in tqdm(dataloader):
        contexts, targets, ctx_polarity, tgt_polarity = batch_preprocess(batch)
        inputs = [(c, placeholder) for c in contexts]
        outputs = scorer.cloze_distribution(inputs)

        # get first token of the targets,
        # does not matter if target is single token
        # targets = [
        #     scorer.tokenizer.decode(
        #         scorer.tokenizer(target, add_special_tokens=False)["input_ids"][0]
        #     )
        #     for target in targets
        # ]

        # Store target indices/scores beforehand
        # might not be present in top-k predictions
        target_indices = [
            scorer.tokenizer(target, add_special_tokens=False)["input_ids"][0]
            for target in targets
        ]
        target_logprobs = [
            outputs[idx][i].detach().cpu().numpy()
            for idx, i in enumerate(target_indices)
        ]

        topk_preds = outputs.topk(topk)
        tokens = topk_preds.indices.detach().cpu().numpy()
        logprobs = topk_preds.values.detach().cpu().numpy()
        tokens = [scorer.tokenizer.convert_ids_to_tokens(t) for t in tokens]
        tokens = cleanup_tokens(
            scorer, tokens
        )  # some models use special chars to denote preceding spaces, remove them

        predictions["context"].extend(contexts)
        predictions["target"].extend(targets)
        predictions["target_logprob"].extend(target_logprobs)
        predictions["tokens"].extend(tokens)
        predictions["logprobs"].extend(logprobs)
        predictions["ctx_polarity"].extend(ctx_polarity)
        predictions["tgt_polarity"].extend(tgt_polarity)

    predictions = pd.DataFrame(predictions)
    return predictions
