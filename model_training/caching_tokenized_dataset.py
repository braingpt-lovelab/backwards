import itertools
import numpy as np
from types import SimpleNamespace
import os
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

import utils


def tokenize(element, tokenizer, args):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=args.chunk_size,
        return_overflowing_tokens=True,
        return_length=True,
    )
    
    output_ids = list(itertools.chain(*outputs["input_ids"]))
    output_mask = list(itertools.chain(*outputs["attention_mask"]))
    
    # Split into chunks:
    output_ids = [output_ids[x:x+args.chunk_size-1] for x in range(0, len(output_ids), args.chunk_size-1)]
    output_mask = [output_mask[x:x+args.chunk_size-1] for x in range(0, len(output_mask), args.chunk_size-1)]

    # Prepend BOS token to each chunk regardless of training direction;
    # if reversed training, first reverse and then prepend BOS,
    # if permuted training, first permute and then prepend BOS.
    if args.reversed_training:
        print("Reversing chunk...")
        output_ids = [chunk[::-1] for chunk in output_ids]
        output_mask = [chunk[::-1] for chunk in output_mask]
    elif args.permuted_training:
        print("Permuting chunk...")
        output_ids_permuted = []
        output_mask_permuted = []
        for chunk in output_ids:
            # Same randomness per chunk
            np.random.seed(args.random_seed)
            permuted_chunk = np.random.permutation(chunk)
            output_ids_permuted.append(list(permuted_chunk))
        for chunk in output_mask:
            # Same randomness per chunk
            np.random.seed(args.random_seed)
            permuted_chunk = np.random.permutation(chunk)
            output_mask_permuted.append(list(permuted_chunk))
        output_ids = output_ids_permuted
        output_mask = output_mask_permuted
    else:
        raise ValueError("Invalid training direction. Must be either reversed or permuted.")
    bos_id = tokenizer.bos_token_id
    output_ids = [[bos_id] + chunk for chunk in output_ids]
    output_mask = [[1] + chunk for chunk in output_mask]
    return {"input_ids": output_ids, "attention_mask": output_mask}


def first_map_then_remove_last(config_fpath):
    # Load config
    args = utils.load_config(config_fpath)
    args = SimpleNamespace(**args)
    print(args)

    # If `args.cache_dir_train` and `args.cache_dir_validation` exist,
    # do nothing.
    if os.path.exists(args.cache_dir_train) \
        and os.path.exists(args.cache_dir_validation):
        print(f"\n\nCache already exists at {args.cache_dir_train} and {args.cache_dir_validation}.")
        print("Exiting...\n\n")
        return None, None

    # Load huggingface dataset
    dataset = load_dataset(args.data_path, cache_dir=args.cache_dir)

    # Load tokenizer
    if args.custom_tokenizer != "None":
        print(f"Load custom tokenizer from {args.custom_tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.custom_tokenizer, cache_dir=args.cache_dir)
    else:
        print(f"Load pretrained tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)

    # Process the dataset (or load from cache if it exists)
    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "args": args},
        batched=True,
        remove_columns=dataset["train"].column_names,
        batch_size=None,  # Load entire dataset at once to handle overflow.
        cache_file_names={
            "train": args.cache_dir_train,
            "validation": args.cache_dir_validation
        }
    )
    
    # Print current dataset sizes
    print(f"Current train size: {len(tokenized_dataset['train'])}")
    print(f"Current validation size: {len(tokenized_dataset['validation'])}")
    
    # Remove the last entry using select() which is more efficient than slicing + from_dict
    for split in ["train", "validation"]:
        split_size = len(tokenized_dataset[split])
        tokenized_dataset[split] = tokenized_dataset[split].select(range(split_size - 1))    
        # Print new dataset sizes
        print(f"New train size: {len(tokenized_dataset['train'])}")
        print(f"New validation size: {len(tokenized_dataset['validation'])}")
    
    # Save the modified dataset to cache (overwriting existing cache)
    for split in ["train", "validation"]:
        cache_path = args.cache_dir_train if split == "train" else args.cache_dir_validation
        # Delete existing cache if it exists
        if os.path.exists(cache_path):
            print(f"Removing existing cache at {cache_path}")
            if os.path.isdir(cache_path):
                import shutil
                shutil.rmtree(cache_path)
            else:
                os.remove(cache_path)
        # Save to disk
        tokenized_dataset[split].save_to_disk(cache_path)
        print(f"Saved modified {split} dataset to {cache_path}")
    
    return tokenized_dataset, tokenizer


def load_from_arrow_dir(config_fpath):
    # Load config
    args = utils.load_config(config_fpath)
    args = SimpleNamespace(**args)
    print(args)

    # Load tokenizer
    if args.custom_tokenizer != "None":
        print(f"Load custom tokenizer from {args.custom_tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.custom_tokenizer, cache_dir=args.cache_dir)
    else:
        print(f"Load pretrained tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)

    train_dataset = datasets.Dataset.load_from_disk(args.cache_dir_train)
    validation_dataset = datasets.Dataset.load_from_disk(args.cache_dir_validation)
    return {
        "train": train_dataset,
        "validation": validation_dataset
    }, tokenizer


if __name__ == "__main__":
    config_fpath = "configs/gpt2_scratch_neuro_tokenizer_bayes_perm.json"  ### _fwd | _rev | _perm
    tokenized_dataset, tokenizer = first_map_then_remove_last(config_fpath)  ### Use once!
    tokenized_dataset, tokenizer = load_from_arrow_dir(config_fpath)  ### comment out ONLY dir exists.

    # Testing
    print(tokenized_dataset)
    for i in range(tokenized_dataset["validation"].num_rows):
        if len(tokenized_dataset["validation"][i]["input_ids"]) != 1024:
            print(i, len(tokenized_dataset["validation"][i]["input_ids"]))
        if i % 100 == 0:
            print(
                i, tokenizer.convert_ids_to_tokens(tokenized_dataset["validation"][i]["input_ids"])[:5]
            )