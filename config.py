from dataclasses import dataclass
from llm_negation.prediction import (
    mlm_distribution,
    next_word_distribution,
    sequence_score,
)


@dataclass
class ModelConfig:
    id: str
    ckpt: str
    type: str
    distribution_function: callable
    sequence_score_function: callable
    scorer_args: dict = {}


MODELS = [
    # MLM models
    ModelConfig("distilbert-base-uncased", "DistilBERT", "MLM", mlm_distribution, sequence_score),
    ModelConfig("bert-base-uncased", "BERT_base", "MLM", mlm_distribution, sequence_score),
    ModelConfig("bert-large-uncased", "BERT_large", "MLM", mlm_distribution, sequence_score),
    ModelConfig("albert/albert-base-v2", "ALBERT_base_v2", "MLM", mlm_distribution, sequence_score),
    ModelConfig("albert/albert-large-v2", "ALBERT_large_v2", "MLM", mlm_distribution, sequence_score),
    ModelConfig("albert/albert-xlarge-v2", "ALBERT_xlarge_v2", "MLM", mlm_distribution, sequence_score),
    ModelConfig("albert/albert-xxlarge-v2", "ALBERT_xxlarge_v2", "MLM", mlm_distribution, sequence_score),
    ModelConfig("FacebookAI/roberta-base", "RoBERTa_base", "MLM", mlm_distribution, sequence_score),
    ModelConfig("FacebookAI/roberta-large", "RoBERTa_large", "MLM", mlm_distribution, sequence_score),
    ModelConfig("answerdotai/ModernBERT-base", "ModernBERT_base", "MLM", mlm_distribution, sequence_score),
    ModelConfig("answerdotai/ModernBERT-large", "ModernBERT_large", "MLM", mlm_distribution, sequence_score),

    # ICLM models
    ModelConfig("google/recurrentgemma-2b-it", "rGemma_2B_it", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("google/recurrentgemma-9b-it", "rGemma_9B_it", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("google/gemma-2b-it", "Gemma_2B_it", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("google/gemma-7b-it", "Gemma_7B_it", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("google/gemma-2-2b-it", "Gemma-2_2B_it", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("google/gemma-2-9b-it", "Gemma-2_9B_it", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2_1B_Instruct", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2_3B_Instruct", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1_8B_Instruct", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", "Qwen-2.5_0.5B_Instruct", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("Qwen/Qwen2.5-1.5B-Instruct", "Qwen-2.5_1.5B_Instruct", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("Qwen/Qwen2.5-3B-Instruct", "Qwen-2.5_3B_Instruct", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("Qwen/Qwen2.5-7B-Instruct", "Qwen-2.5_7B_Instruct", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("allenai/OLMo-2-1124-7B-SFT", "OLMo-2_1124_7B_SFT", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("allenai/OLMo-2-1124-7B-DPO", "OLMo-2_1124_7B_DPO", "ICLM", next_word_distribution, sequence_score),
    ModelConfig("allenai/OLMo-2-1124-7B-Instruct", "OLMo-2_1124_7B_Instruct", "ICLM", next_word_distribution, sequence_score),
    
    # CLM models
    ModelConfig("openai-community/gpt2", "GPT2", "CLM", next_word_distribution, sequence_score),
    ModelConfig("openai-community/gpt2-medium", "GPT2_medium", "CLM", next_word_distribution, sequence_score),
    ModelConfig("openai-community/gpt2-large", "GPT2_large", "CLM", next_word_distribution, sequence_score),
    ModelConfig("openai-community/gpt2-xl", "GPT2_xl", "CLM", next_word_distribution, sequence_score),
    ModelConfig("google/recurrentgemma-2b", "rGemma_2B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("google/recurrentgemma-9b", "rGemma_9B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("google/gemma-2b", "Gemma_2B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("google/gemma-7b", "Gemma_7B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("google/gemma-2-2b", "Gemma-2_2B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("google/gemma-2-9b", "Gemma-2_9B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("meta-llama/Llama-3.2-1B", "Llama-3.2_1B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("meta-llama/Llama-3.2-3B", "Llama-3.2_3B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("meta-llama/Llama-3.1-8B", "Llama-3.1_8B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("Qwen/Qwen2.5-0.5B", "Qwen-2.5_0.5B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("Qwen/Qwen2.5-1.5B", "Qwen-2.5_1.5B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("Qwen/Qwen2.5-3B", "Qwen-2.5_3B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("Qwen/Qwen2.5-7B", "Qwen-2.5_7B", "CLM", next_word_distribution, sequence_score),
    ModelConfig("EleutherAI/pythia-70m-deduped", "Pythia_70M_deduped", "CLM", next_word_distribution, sequence_score),
    ModelConfig("EleutherAI/pythia-160m-deduped", "Pythia_160M_deduped", "CLM", next_word_distribution, sequence_score),
    ModelConfig("EleutherAI/pythia-410m-deduped", "Pythia_410M_deduped", "CLM", next_word_distribution, sequence_score),
    ModelConfig("EleutherAI/pythia-1b-deduped", "Pythia_1B_deduped", "CLM", next_word_distribution, sequence_score),
    ModelConfig("EleutherAI/pythia-1.4b-deduped", "Pythia_1.4B_deduped", "CLM", next_word_distribution, sequence_score),
    ModelConfig("EleutherAI/pythia-2.8b-deduped", "Pythia_2.8B_deduped", "CLM", next_word_distribution, sequence_score),
    ModelConfig("EleutherAI/pythia-6.9b-deduped", "Pythia_6.9B_deduped", "CLM", next_word_distribution, sequence_score),
    ModelConfig("allenai/OLMo-2-1124-7B", "OLMo-2_1124_7B", "CLM", next_word_distribution, sequence_score),
    
    # SEQ2SEQ models
    ModelConfig("google-t5/t5-small", "T5_small", "SEQ2SEQ", mlm_distribution, sequence_score),
    ModelConfig("google-t5/t5-base", "T5_base", "SEQ2SEQ", mlm_distribution, sequence_score),
    ModelConfig("google-t5/t5-large", "T5_large", "SEQ2SEQ", mlm_distribution, sequence_score),
    ModelConfig("google-t5/t5-3b", "T5_3B", "SEQ2SEQ", mlm_distribution, sequence_score),
    
    # MAMBA models
    ModelConfig("state-spaces/mamba-130m", "Mamba_130M", "MAMBA", next_word_distribution, sequence_score, scorer_args={"tokenizer": "EleutherAI/gpt-neox-20b"}),
    ModelConfig("state-spaces/mamba-790m", "Mamba_790M", "MAMBA", next_word_distribution, sequence_score, scorer_args={"tokenizer": "EleutherAI/gpt-neox-20b"}),
    ModelConfig("state-spaces/mamba-2.8b", "Mamba_2.8B", "MAMBA", next_word_distribution, sequence_score, scorer_args={"tokenizer": "EleutherAI/gpt-neox-20b"}),
    ModelConfig("state-spaces/mamba2-130m", "Mamba2_130M", "MAMBA", next_word_distribution, sequence_score, scorer_args={"tokenizer": "EleutherAI/gpt-neox-20b"}),
    ModelConfig("state-spaces/mamba2-780m", "Mamba2_780M", "MAMBA", next_word_distribution, sequence_score, scorer_args={"tokenizer": "EleutherAI/gpt-neox-20b"}),
    ModelConfig("state-spaces/mamba2-1.3b", "Mamba2_1.3B", "MAMBA", next_word_distribution, sequence_score, scorer_args={"tokenizer": "EleutherAI/gpt-neox-20b"}),
    ModelConfig("state-spaces/mamba2-2.7b", "Mamba2_2.7B", "MAMBA", next_word_distribution, sequence_score, scorer_args={"tokenizer": "EleutherAI/gpt-neox-20b"}),
]

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
