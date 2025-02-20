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
    wordnet_prefix_word: str,
    prefix: str,
    determiner: bool,
) -> list[tuple[str, str, str, str]]:
    data = []
    context_aff = " ".join(context_aff.split()[:-1])
    context_neg = " ".join(context_neg.split()[:-1])

    if wordnet_prefix_word and isinstance(wordnet_prefix_word, str):
        wn_prefix = get_wordnet_prefix(wordnet_prefix_word)
    elif wordnet_prefix_word and isinstance(wordnet_prefix_word, tuple):
        wn_prefix = get_wordnet_prefix(wordnet_prefix_word[0]) + get_wordnet_prefix(
            wordnet_prefix_word[1]
        )
    else:
        wn_prefix = ""

    if determiner:
        aff_det = get_determiner(target_aff)
        neg_det = get_determiner(target_neg)
        data.append(
            (f"{wn_prefix}{prefix}{context_aff} {aff_det}", target_aff, "aff", "aff")
        )
        data.append(
            (f"{wn_prefix}{prefix}{context_aff} {neg_det}", target_neg, "aff", "neg")
        )
        data.append(
            (f"{wn_prefix}{prefix}{context_neg} {neg_det}", target_neg, "neg", "neg")
        )
        data.append(
            (f"{wn_prefix}{prefix}{context_neg} {aff_det}", target_aff, "neg", "aff")
        )
    else:
        data.append((f"{wn_prefix}{prefix}{context_aff}", target_aff, "aff", "aff"))
        data.append((f"{wn_prefix}{prefix}{context_aff}", target_neg, "aff", "neg"))
        data.append((f"{wn_prefix}{prefix}{context_neg}", target_neg, "neg", "neg"))
        data.append((f"{wn_prefix}{prefix}{context_neg}", target_aff, "neg", "aff"))
    return data


def prepare_dataset_neg(
    dataset, wordnet_prefix_word: str = "", prefix: str = "", determiner: bool = True
):
    temp = wordnet_prefix_word
    prepared_dataset = []
    for _, row in dataset.iterrows():
        if temp == "aff":
            wordnet_prefix_word = row["target_aff"]
        elif temp == "neg":
            wordnet_prefix_word = row["target_neg"]
        elif temp == "both":
            wordnet_prefix_word = (row["target_aff"], row["target_neg"])
        elif temp == "rboth":
            wordnet_prefix_word = (row["target_neg"], row["target_aff"])

        data = prepare_data_neg(
            row["context_aff"],
            row["context_neg"],
            row["target_aff"],
            row["target_neg"],
            wordnet_prefix_word=wordnet_prefix_word,
            prefix=prefix,
            determiner=determiner,
        )
        prepared_dataset.extend(data)

    return prepared_dataset
