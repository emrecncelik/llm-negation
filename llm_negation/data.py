import pandas as pd


def get_determiner(word: str, vowels="aeiou") -> str:
    if word[0] in vowels:
        return "an"
    else:
        return "a"


def apply_chat_template(
    tokenizer,
    user_message: str,
    assistant_message: str,
):
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]
    messages = tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )
    return messages


def prepare_prompt(
    tokenizer,
    context: str,
    target: str,
    prompt_template: str,
    assistant_message_template: str,
) -> str:
    if assistant_message_template:
        user_message = prompt_template.format(
            context=context,
            target=target,
            determiner=get_determiner(target),
        )
        assistant_message = assistant_message_template.format(
            context=context, target=target, determiner=get_determiner(target)
        )
        prompt = apply_chat_template(tokenizer, user_message, assistant_message)
    else:
        prompt = prompt_template.format(
            context=context, target=target, determiner=get_determiner(target)
        )

    return prompt


def prepare_negation_data(
    tokenizer,
    context_aff: str,
    context_neg: str,
    target_aff: str,
    target_neg: str,
    prompt_template: str,
    assistant_message_template: str,
) -> list[tuple[str, str, str, str]]:
    if "{determiner}" in prompt_template:
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
            prepare_prompt(
                tokenizer=tokenizer,
                context=context,
                target=target,
                prompt_template=prompt_template,
                assistant_message_template=assistant_message_template,
            ),
            target,
            ctx_polarity,
            tgt_polarity,
        )
        for context, target, ctx_polarity, tgt_polarity in combinations
    ]


def prepare_negation_dataset(
    tokenizer,
    dataset: pd.DataFrame,
    prompt_template: str,
    assistant_message_template: str,
):
    prepared_dataset = []
    for _, row in dataset.iterrows():
        data = prepare_negation_data(
            tokenizer=tokenizer,
            context_aff=row["context_aff"],
            context_neg=row["context_neg"],
            target_aff=row["target_aff"],
            target_neg=row["target_neg"],
            prompt_template=prompt_template,
            assistant_message_template=assistant_message_template,
        )
        prepared_dataset.extend(data)

    return prepared_dataset
