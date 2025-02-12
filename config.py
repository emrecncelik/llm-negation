# "ai21labs/AI21-Jamba-1.5-Mini",
# "ai21labs/AI21-Jamba-1.5-Large",


def get_model_type(model_name: str) -> str:
    for model_type, model_list in MODELS.items():
        if model_name in model_list:
            return model_type
    raise ValueError(f"Model {model_name} not found in config.MODELS.")


MODELS = {
    "MLM": [
        "distilbert-base-uncased",
        "bert-base-uncased",
        "bert-large-uncased",
        "albert/albert-base-v2",
        "albert/albert-large-v2",
        "albert/albert-xlarge-v2",
        "albert/albert-xxlarge-v2",
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large",
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
    ],
    "CLM": [
        "openai-community/gpt2",
        "openai-community/gpt2-medium",
        # "openai-community/gpt2-large",
        # "openai-community/gpt2-xl",
        # recurrent gemma may require different configuration
        "google/recurrentgemma-2b",
        # "google/recurrentgemma-9b",
        "google/gemma-2b",
        # "google/gemma-7b",
        "google/gemma-2-2b",
        # "google/gemma-2-9b",
        # "google/gemma-2-27b",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        # "meta-llama/Llama-3.1-8B",
        # "meta-llama/Llama-3.1-70B",
        # "meta-llama/Llama-3.1-405B",
        "EleutherAI/pythia-70m-deduped",
        "EleutherAI/pythia-160m-deduped",
        "EleutherAI/pythia-410m-deduped",
        "EleutherAI/pythia-1b-deduped",
        "EleutherAI/pythia-1.4b-deduped",
        # "EleutherAI/pythia-2.8b-deduped",
        # "EleutherAI/pythia-6.9b-deduped",
        # "EleutherAI/pythia-12b-deduped",
    ],
    "SEQ2SEQ": [
        "google-t5/t5-small",
        "google-t5/t5-base",
        "google-t5/t5-large",
        # flan may require different configuration
        "google/flan-t5-small",
        # "google/flan-t5-base",
        # "google/flan-t5-large",
        # "google/flan-t5-xl",
        # "google/flan-t5-xxl",
    ],
    "MAMBA": [
        # "state-spaces/mamba-130m-hf",
        "state-spaces/mamba-790m-hf",
        # "state-spaces/mamba-2.8b-hf",
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
