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
Run the saved models on a dataset (train or validation) to get distribution of PPLs.
"""

def collate_fn(batch):
    input_ids = [sample["input_ids"] for sample in batch]
    attention_masks = [sample["attention_mask"] for sample in batch]
    labels = input_ids

    # Convert to tensor
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks, 
        "labels": labels,
    }


def evaluate(LLM, dataloader, tokenizer):
    all_batches_ppl = []
    all_batches_tokens_n_ppls = {}
    for i, batch in enumerate(dataloader):
        # # Plot the first and last 5 tokens
        # print(f"\nBatch {i}")
        # print(f"{tokenizer.convert_ids_to_tokens(batch['input_ids'][0][:5])}")
        # print(f"{tokenizer.convert_ids_to_tokens(batch['input_ids'][0][-5:])}")
        with torch.cuda.amp.autocast():
            labels = batch["labels"].to(LLM.device)
            batch = {
                "input_ids": batch["input_ids"].to(LLM.device), 
                "attention_mask": batch["attention_mask"].to(LLM.device)
            }
            logits = LLM(**batch).logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

            # Compute loss
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            mask = torch.zeros_like(logits)
            bos_token_id = tokenizer.bos_token_id
            mask[:, bos_token_id] = float('-inf')
            masked_logits = logits + mask

            log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
            true_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).float()
            loss = -true_log_probs.mean()
            ppl = math.exp(loss.item())
            # print(f"PPL: {ppl}")
            all_batches_ppl.append(ppl)
        
        # Save each batch's tokens and ppls for further text analysis
        all_batches_tokens_n_ppls[i] = {
            "tokens": tokenizer.convert_ids_to_tokens(batch["input_ids"][0]),
            "logprobs": true_log_probs.tolist(),
            "ppl": ppl,
        }

    return all_batches_ppl, all_batches_tokens_n_ppls


def main(llm, dataset_type, sample_percentage):
    result_fpath = os.path.join(results_dir, f"all_batches_ppl_{dataset_type}.npy")
    if not os.path.exists(result_fpath):
        config_fpath = os.path.join(configs_dir, f"{llm}.json")
        with open(config_fpath, "r") as f:
            args = json.load(f)
        
        args["cache_dir_train"] = os.path.join(reference_dir, args["cache_dir_train"])
        args["cache_dir_validation"] = os.path.join(reference_dir, args["cache_dir_validation"])
        if "backwards" in llm:
            tokenizer_dir = "cache/gpt2_neuro_tokenizer_backwards"
        else:
            tokenizer_dir = "cache/gpt2_neuro_tokenizer"
        args["custom_tokenizer"] = os.path.join(reference_dir, tokenizer_dir)
        args["spt"] = ""
        args["batch_size"] = 1
        args = argparse.Namespace(**args)

        # Load dataset
        if dataset_type == "train":
            dataset = datasets.Dataset.load_from_disk(args.cache_dir_train)
            if sample_percentage < 1.0:
                dataset = dataset.shuffle(seed=42).select(
                    range(int(len(dataset) * sample_percentage))
                )
        else:
            dataset = datasets.Dataset.load_from_disk(args.cache_dir_validation)

        print(f"Loading {len(dataset)} samples for {dataset_type}")
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )

        # Load model
        LLM, tokenizer = model_utils.load_model_and_tokenizer(llm)

        # Run model on data to get ppl
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        with torch.no_grad():
            LLM.eval()
            all_batches_ppl, all_batches_tokens_n_ppls = evaluate(LLM, dataloader, tokenizer)
        
        # Save results
        np.save(
            os.path.join(results_dir, f"all_batches_ppl_{dataset_type}.npy"),
            np.array(all_batches_ppl)
        )
    else:
        print(f"Results already exist at {result_fpath}. Skipping evaluation.")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str, default="True")
    parser.add_argument("--dataset_type", type=str, choices=["train", "validation"], default="validation")
    parser.add_argument("--sample_percentage", type=float, default=0.02, help="Percentage of train dataset to sample (only applicable if dataset_type is 'train')")

    args = parser.parse_args()

    use_human_abstract = args.use_human_abstract.lower() == "true"
    dataset_type = args.dataset_type
    sample_percentage = args.sample_percentage

    model_list = model_utils.model_list
    llms = [llm for llm_family in model_list for llm in model_list[llm_family]]

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

        main(llm, dataset_type, sample_percentage)
