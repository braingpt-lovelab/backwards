import os
import math
import json
import argparse
import itertools

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils import model_utils

"""
Run the saved models on a validation data to get distribution of PPLs.
"""

def tokenize(element, tokenizer, args):
    if args.spt != "":
        print(f"Adjust max_length to {args.chunk_size - 1} to add special tokens")
        special_token_id = tokenizer.convert_tokens_to_ids(args.spt)
        max_length = args.chunk_size - 1
    else:
        print("Not using special token")
        max_length = args.chunk_size

    # Reverse text if testing backwards trained models
    if "backwards" in args.custom_tokenizer:
        element = {"text": [x[::-1] for x in element["text"]]}

    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    # Add special token id to the end of every chunk
    if args.spt != "":
        print(f"Add special token: {args.spt} to separate documents")
        for i in range(len(outputs["input_ids"])):
            outputs["input_ids"][i].append(special_token_id)
            outputs["attention_mask"][i].append(1)

    output_ids = list(itertools.chain(*outputs["input_ids"]))
    output_mask = list(itertools.chain(*outputs["attention_mask"]))
    output_ids = [output_ids[x:x+args.chunk_size] for x in range(0, len(output_ids), args.chunk_size)]
    output_mask = [output_mask[x:x+args.chunk_size] for x in range(0, len(output_mask), args.chunk_size)]
    return {"input_ids": output_ids, "attention_mask": output_mask}


def collate_fn(batch):
    input_ids = [sample["input_ids"] for sample in batch]
    attention_masks = [sample["attention_mask"] for sample in batch]
    labels = pad_sequence(input_ids, batch_first=True, padding_value=-1)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids,  
        "attention_mask": attention_masks,  
        "labels": labels,
    }


def evaluate(args, LLM, valid_dataloader, criterion):
    all_batches_ppl = []
    for i, batch in enumerate(valid_dataloader):
        print(f"Batch {i}/{len(valid_dataloader)}")
        with torch.cuda.amp.autocast():
            labels = batch["labels"].to(LLM.device)
            batch = {
                "input_ids": batch["input_ids"].to(LLM.device),
                "attention_mask": batch["attention_mask"].to(LLM.device),
            }
            logits = LLM(**batch).logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            ppl = math.exp(loss.item())
            all_batches_ppl.append(ppl)

    return all_batches_ppl


def main(llm):
    config_fpath = os.path.join(configs_dir, f"{llm}.json")
    with open(config_fpath, "r") as f:
        args = json.load(f)

    # Override a few things using full path.
    args["cache_dir"] = os.path.join(reference_dir, args["cache_dir"])
    if "backwards" in llm:
        tokenizer_dir = "cache/gpt2_neuro_tokenizer_backwards"
    else:
        tokenizer_dir = "cache/gpt2_neuro_tokenizer"
    args["custom_tokenizer"] = os.path.join(reference_dir, tokenizer_dir)
    args["spt"] = ""
    args["batch_size"] = 1  # for building distribution of ppl
    args = argparse.Namespace(**args)

    # Load validation data
    dataset = load_dataset(args.data_path, cache_dir=args.cache_dir)
    del dataset['train']

    tokenizer = AutoTokenizer.from_pretrained(args.custom_tokenizer, cache_dir=args.cache_dir)
    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "args": args},
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    print("Loading {} samples for validation".format(len(tokenized_dataset["validation"])))
    tokenized_dataset.set_format("torch")
    valid_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=1,
        collate_fn=collate_fn,
    )

    # Load model
    LLM, _ = model_utils.load_model_and_tokenizer(llm)

    # Run model on data to get ppl
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    with torch.no_grad():
        LLM.eval()
        all_batches_ppl = evaluate(args, LLM, valid_dataloader, criterion)
    
    # Save per llm all batches ppl
    np.save(
        os.path.join(results_dir, f"all_batches_ppl_validation.npy"),
        np.array(all_batches_ppl)
    )
        

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str, default="True")

    if parser.parse_args().use_human_abstract == "True":
        use_human_abstract = True
    else:
        use_human_abstract = False

    llms = [
        # "gpt2_scratch_neuro_tokenizer",
        # "gpt2_scratch_neuro_tokenizer_backwards",
        "gpt2-medium_scratch_neuro_tokenizer",
        "gpt2-medium_scratch_neuro_tokenizer_backwards",
        "gpt2-large_scratch_neuro_tokenizer",
        "gpt2-large_scratch_neuro_tokenizer_backwards"
    ]

    # TODO: eventually we will move reference to backwards/ out of matching_experts
    reference_dir = "/home/ken/projects/matching_experts/model_training"
    configs_dir = "/home/ken/projects/backwards/model_training/configs"

    for llm in llms:
        if use_human_abstract:
            type_of_abstract = 'human_abstracts'
        else:
            type_of_abstract = 'llm_abstracts'
        results_dir = f"model_results/{llm.replace('/', '--')}/{type_of_abstract}"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            results_dir = "model_results"

        main(llm)