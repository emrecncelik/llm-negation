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
    context_aff: str,
    context_neg: str,
    target_aff: str,
    target_neg: str,
    determiner: bool,
    prompt_format: str,
) -> list[tuple[str, str, str, str]]:
    if determiner:
        context_aff = " ".join(context_aff.split()[:-1])
        context_neg = " ".join(context_neg.split()[:-1])

    combinations = [
        (context_aff, target_aff, "aff", "aff"),
        (context_aff, target_neg, "aff", "neg"),
        (context_neg, target_neg, "neg", "neg"),
        (context_neg, target_aff, "neg", "aff"),
    ]

    return [
        (
            prepare_prompt(prompt_format, context, target, determiner),
            target,
            ctx_polarity,
            tgt_polarity,
        )
        for context, target, ctx_polarity, tgt_polarity in combinations
    ]


def prepare_dataset_neg(
    dataset,
    prompt_format: str = "{context} {determiner}",
    determiner: bool = True,
):
    prepared_dataset = []
    for _, row in dataset.iterrows():

        data = prepare_data_neg(
            row["context_aff"],
            row["context_neg"],
            row["target_aff"],
            row["target_neg"],
            determiner=determiner,
            prompt_format=prompt_format,
        )
        prepared_dataset.extend(data)

    return prepared_dataset
