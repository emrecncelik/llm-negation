from nltk.corpus import wordnet as wn


def get_determiner(word: str, vowels="aeiou") -> str:
    if word[0] in vowels:
        return "an"
    else:
        return "a"


def prepare_prompt(
    prompt_format: str,
    context: str,
    target: str,
    determiner: bool,
) -> str:
    if determiner:
        return prompt_format.format(
            context=context, target=target, determiner=get_determiner(target)
        )
    else:
        return prompt_format.format(context=context, target=target)


def prepare_data_neg(
    prompt_format: str,
    context_aff: str,
    context_neg: str,
    target_aff: str,
    target_neg: str,
    determiner: bool,
) -> list[tuple[str, str, str, str]]:
    data = []
    context_aff = " ".join(context_aff.split()[:-1])
    context_neg = " ".join(context_neg.split()[:-1])

    if determiner:
        data.append(prepare_prompt(prompt_format, context_aff, target_aff, True), target_aff, "aff", "aff")
        data.append(prepare_prompt(prompt_format, context_aff, target_neg, True), target_neg, "aff", "neg")
        data.append(prepare_prompt(prompt_format, context_neg, target_neg, True), target_neg, "neg", "neg")
        data.append(prepare_prompt(prompt_format, context_neg, target_aff, True), target_aff, "neg", "aff")
    else:
        data.append(prepare_prompt(prompt_format, context_aff, target_aff, False), target_aff, "aff", "aff")
        data.append(prepare_prompt(prompt_format, context_aff, target_neg, False), target_neg, "aff", "neg")
        data.append(prepare_prompt(prompt_format, context_neg, target_neg, False), target_neg, "neg", "neg")
        data.append(prepare_prompt(prompt_format, context_neg, target_aff, False), target_aff, "neg", "aff")
    return data


def prepare_dataset_neg(
    dataset,
    prompt_format: str = "{context} {determiner}",
    determiner: bool = True,
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
            determiner=determiner,
        )
        prepared_dataset.extend(data)

    return prepared_dataset
