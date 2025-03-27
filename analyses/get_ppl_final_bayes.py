import os
import math
import json
import argparse
import itertools

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils import model_utils

"""
Run the saved models on a validation data to get distribution of PPLs.
"""

def collate_fn(batch):
    input_ids = [sample["input_ids"] for sample in batch]
    attention_masks = [sample["attention_mask"] for sample in batch]
    labels = input_ids

    # Convert to tensor
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    # No need to pad anything; handled by `caching_tokenized_dataset.py`
    # If got error means we made a mistake somehow...
    # labels = pad_sequence(input_ids, batch_first=True, padding_value=-1)
    # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    # attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks, 
        "labels": labels,
    }


def evaluate(args, LLM, valid_dataloader, criterion, tokenizer):
    all_batches_ppl = []
    for i, batch in enumerate(valid_dataloader):
        print(f"Batch {i}/{len(valid_dataloader)}")
        with torch.cuda.amp.autocast():
            labels = batch["labels"]
            batch = {
                "input_ids": batch["input_ids"], 
                "attention_mask": batch["attention_mask"]
            }
            logits = LLM(**batch).logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

            # Compute loss
            # - flatten logits and labels
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            # - logsoftmax must ignore the BOS column
            mask = torch.zeros_like(logits)
            bos_token_id = tokenizer.bos_token_id
            mask[:, bos_token_id] = float('-inf')  # Set the BOS column logits to -inf
            masked_logits = logits + mask

            # Softmax and compute loss
            log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
            true_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).float()
            loss = -true_log_probs.mean()
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
    tokenized_dataset = {
        "validation": datasets.Dataset.load_from_disk(args.cache_dir_validation),
    }
    print("Loading {} samples for validation".format(len(tokenized_dataset["validation"])))
    tokenized_dataset.set_format("torch")
    valid_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.batch_size,
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
        "gpt2_scratch_neuro_tokenizer_bayes_fwd",
        "gpt2_scratch_neuro_tokenizer_bayes_rev",
    ]

    reference_dir = "/home/ken/projects/backwards/model_training"
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