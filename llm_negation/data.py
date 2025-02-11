from nltk.corpus import wordnet as wn


def get_determiner(word: str, vowels="aeiou") -> str:
    if word[0] in vowels:
        return "an"
    else:
        return "a"


def get_wordnet_prefix(word):
    definition = wn.synsets(word)[0].definition()
    return f"{get_determiner(word)} {word} is {definition}. "


def prepare_data_neg(
    context_aff: str,
    context_neg: str,
    target_aff: str,
    target_neg: str,
    wordnet_prefix: str,
    determiner: bool,
) -> list[tuple[str, str, str, str]]:
    data = []
    context_aff = " ".join(context_aff.split()[:-1])
    context_neg = " ".join(context_neg.split()[:-1])

    if wordnet_prefix:
        prefix = get_wordnet_prefix(target_aff)
    else:
        prefix = ""

    if determiner:
        aff_det = get_determiner(target_aff)
        neg_det = get_determiner(target_neg)
        data.append((f"{prefix}{context_aff} {aff_det}", target_aff, "aff", "aff"))
        data.append((f"{prefix}{context_aff} {neg_det}", target_neg, "aff", "neg"))
        data.append((f"{prefix}{context_neg} {neg_det}", target_neg, "neg", "neg"))
        data.append((f"{prefix}{context_neg} {aff_det}", target_aff, "neg", "aff"))
    else:
        data.append((f"{prefix}{context_aff}", target_aff, "aff", "aff"))
        data.append((f"{prefix}{context_aff}", target_neg, "aff", "neg"))
        data.append((f"{prefix}{context_neg}", target_neg, "neg", "neg"))
        data.append((f"{prefix}{context_neg}", target_aff, "neg", "aff"))
    return data


def prepare_dataset_neg(dataset, wordnet_prefix: bool = False, determiner: bool = True):
    prepared_dataset = []
    for _, row in dataset.iterrows():
        data = prepare_data_neg(
            row["context_aff"],
            row["context_neg"],
            row["target_aff"],
            row["target_neg"],
            wordnet_prefix=wordnet_prefix,
            determiner=determiner,
        )
        prepared_dataset.extend(data)

    return prepared_dataset
