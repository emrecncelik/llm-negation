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
    "CLM": [],
    "SEQ2SEQ": [],
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
