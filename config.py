from minicons import scorer
from llm_negation.prediction import (
    mlm_distribution,
    next_word_distribution,
    conditional_score,
)

# "ai21labs/AI21-Jamba-1.5-Mini",
# "ai21labs/AI21-Jamba-1.5-Large",


def get_model_type(model_name: str) -> str:
    for model_type, model_list in MODELS.items():
        if model_name in model_list:
            return model_type
    raise ValueError(f"Model {model_name} not found in config.MODELS.")


SCORER_CONFIG = {
    "CLM": {
        "scorer_class": scorer.IncrementalLMScorer,
        "distribution_func": next_word_distribution,
        "conditional_func": conditional_score,
        "extra_args": {},
    },
    "MLM": {
        "scorer_class": scorer.MaskedLMScorer,
        "distribution_func": mlm_distribution,
        "conditional_func": conditional_score,
        "extra_args": {},
    },
    "MAMBA": {
        "scorer_class": scorer.MambaScorer,
        "distribution_func": next_word_distribution,
        "conditional_func": conditional_score,
        "extra_args": {"tokenizer": "EleutherAI/gpt-neox-20b"},
    },
    "SEQ2SEQ": {
        "scorer_class": scorer.Seq2SeqScorer,
        "distribution_func": mlm_distribution,
        "conditional_func": conditional_score,
        "extra_args": {},
    },
}


MODELS = {
    "MLM": [
        "distilbert-base-uncased",
        "bert-base-uncased",
        "bert-large-uncased",
        "albert/albert-base-v2",
        "albert/albert-large-v2",
        # "albert/albert-xlarge-v2",
        # "albert/albert-xxlarge-v2",
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large",
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
    ],
    "CLM": [
        "openai-community/gpt2",
        # "openai-community/gpt2-medium",
        # "openai-community/gpt2-large",
        # "openai-community/gpt2-xl",
        # "google/recurrentgemma-2b",
        # "google/recurrentgemma-2b-it",
        # "google/recurrentgemma-9b",
        # "google/recurrentgemma-9b-it",
        # "google/gemma-2b",
        # "google/gemma-2b-it",
        # "google/gemma-7b",
        # "google/gemma-7b-it",
        # "google/gemma-2-2b",
        # "google/gemma-2-2b-it",
        # "google/gemma-2-9b",
        # "google/gemma-2-9b-it",
        # "google/gemma-2-27b",
        # "google/gemma-2-27b-it",
        # "meta-llama/Llama-3.2-1B",
        # "meta-llama/Llama-3.2-1B-Instruct",
        # "meta-llama/Llama-3.2-3B",
        # "meta-llama/Llama-3.2-3B-Instruct",
        # "meta-llama/Llama-3.1-8B",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "meta-llama/Llama-3.1-70B",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "Qwen/Qwen2.5-0.5B",
        # "Qwen/Qwen2.5-0.5B-Instruct",
        # "Qwen/Qwen2.5-1.5B",
        # "Qwen/Qwen2.5-1.5B-Instruct",
        # "Qwen/Qwen2.5-3B",
        # "Qwen/Qwen2.5-3B-Instruct",
        # "Qwen/Qwen2.5-7B",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "Qwen/Qwen2.5-14B",
        # "Qwen/Qwen2.5-14B-Instruct",
        # "Qwen/Qwen2.5-32B",
        # "Qwen/Qwen2.5-32B-Instruct",
        # "Qwen/Qwen2.5-72B",
        # "Qwen/Qwen2.5-72B-Instruct",
        # "EleutherAI/pythia-70m-deduped",
        # "EleutherAI/pythia-160m-deduped",
        # "EleutherAI/pythia-410m-deduped",
        # "EleutherAI/pythia-1b-deduped",
        # "EleutherAI/pythia-1.4b-deduped",
        # "EleutherAI/pythia-2.8b-deduped",
        # "EleutherAI/pythia-6.9b-deduped",
        # "EleutherAI/pythia-12b-deduped",
    ],
    "SEQ2SEQ": [
        # "google-t5/t5-small",
        # "google-t5/t5-base",
        # "google-t5/t5-large",
        # "google/flan-t5-small",
        # "google/flan-t5-base",
        # "google/flan-t5-large",
        # "google/flan-t5-xl",
        # "google/flan-t5-xxl",
    ],
    "MAMBA": [
        # "state-spaces/mamba-130m",
        # "state-spaces/mamba-790m",
        # "state-spaces/mamba-2.8b",
        # "state-spaces/mamba2-130m",
        # "state-spaces/mamba2-780m",
        # "state-spaces/mamba2-1.3b",
        # "state-spaces/mamba2-2.7b",
    ],
}

DATASETS = {
    "neg1500gen": {
        "filename": "NEG-1500-SIMP-GEN.txt",
        "url": "https://huggingface.co/datasets/text-machine-lab/NEG-1500-SIMP-GEN/raw/main/NEG-1500-SIMP-GEN.txt",
        "format": True,
    },
    "neg1500temp": {
        "filename": "NEG-1500-SIMP-TEMP.txt",
        "url": "https://huggingface.co/datasets/text-machine-lab/NEG-1500-SIMP-TEMP/raw/main/NEG-1500-SIMP-TEMP.txt",
        "format": True,
    },
    "neg136nat": {
        "filename": "NEG-136-NAT.tsv",
        "url": "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/NEG-136/NEG-136-NAT.tsv",
        "format": False,
    },
    "neg136simp": {
        "filename": "NEG-136-SIMP.tsv",
        "url": "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/NEG-136/NEG-136-SIMP.tsv",
        "format": False,
    },
}
