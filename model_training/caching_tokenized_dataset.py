from types import SimpleNamespace
from datasets import load_dataset
from transformers import AutoTokenizer

import utils
from train_bayes import tokenize


def main(config_fpath):
    # Load config
    args = utils.load_config(config_fpath)
    args = SimpleNamespace(**args)
    print(args)

    # Load huggingface dataset
    dataset = load_dataset(args.data_path, cache_dir=args.cache_dir)

    # Load tokenizer
    if args.custom_tokenizer != "None":
        print(f"Load custom tokenizer from {args.custom_tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.custom_tokenizer, cache_dir=args.cache_dir)
    else:
        print(f"Load pretrained tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)

    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "args": args},
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=None,  # Load entire dataset at once to handle overflow.
    )
    return tokenized_dataset


if __name__ == "__main__":
    config_fpath = "configs/gpt2_scratch_neuro_tokenizer_bayes_rev.json"
    tokenized_dataset = main(config_fpath)
    
    for i in range(tokenized_dataset["validation"].num_rows):
        if len(tokenized_dataset["validation"][i]["input_ids"]) != 1024:
            print(i, len(tokenized_dataset["validation"][i]["input_ids"]))