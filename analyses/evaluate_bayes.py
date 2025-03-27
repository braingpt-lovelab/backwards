import os
import random
import argparse
import math
import pickle
import time
import json
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import SchedulerType, AdamW, get_scheduler
import datasets
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig

import utils


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


def main(rank, args, world_size):
    print(f"rank: {rank}")

    tokenized_dataset = {
        "validation": datasets.Dataset.load_from_disk(args.cache_dir_validation),
    }

    print(tokenized_dataset)
    print("Loading {} samples for validation".format(len(tokenized_dataset["validation"])))

    valid_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    model_fpath = "gpt2_scratch_neuro_tokenizer_bayes_fwd"
    model_fpath = f"/home/ken/projects/backwards/model_training/exp/{model_fpath}/checkpoint.4"
    print("Loading GPT2 model from", model_fpath)
    load_in_8bit = False
    torch_dtype = torch.float16
    LLM = transformers.GPT2LMHeadModel.from_pretrained(
        model_fpath,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )

    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        model_fpath,    
    )

    LLM, valid_dataloader = accelerator.prepare(
        LLM, valid_dataloader
    )

    LLM.eval()
    with torch.no_grad():
        print(f"rank: {accelerator.process_index}, len(valid_dataloader): {len(valid_dataloader)}")
        val_loss = evaluate(args, LLM, valid_dataloader, tokenizer)
        print(f"rank: {accelerator.process_index}, val_loss: {val_loss}, val_ppl: {math.exp(val_loss)}")
        torch.distributed.reduce(val_loss, 0)
        val_loss = val_loss / world_size

        if accelerator.is_main_process:
            val_ppl = math.exp(val_loss)
            print(f"main | val_loss: {val_loss}, validation PPL: {val_ppl}")


def evaluate(args, LLM, valid_dataloader, tokenizer):
    total_tokens = 0
    total_loss = 0.
    for i, batch in enumerate(valid_dataloader):
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
            
            ntokens = (batch["attention_mask"][:, 1:] == 1).sum()
            print(f"batch: {i}, rank: {accelerator.process_index}, ntokens: {ntokens}, loss: {loss}")
            # decode each sample and print first 5 tokens
            for sample_idx, sample in enumerate(batch["input_ids"]):
                decoded_sample = tokenizer.decode(sample[:5], skip_special_tokens=True)
                print(f"rank: {accelerator.process_index}, sample_idx: {sample_idx}, decoded_sample: {decoded_sample}")
            total_tokens += ntokens
            total_loss += loss * ntokens
    
    print(f"rank: {accelerator.process_index}, total_loss: {total_loss}, total_tokens: {total_tokens}")
    return total_loss / total_tokens


if __name__ == "__main__":
    ## Parameter groups
    parser = argparse.ArgumentParser(description="BrainlessGPT")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./hf_models",
        help="Path to the model file",
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./cache',
        help='Path to the cache directory'
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./hf_models",
        help="Path to the train data file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4096,
        help="maximum number of tokens in each sample"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default='./exp/clip_vlm',
        help="Path to the output dir",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="log interval",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Saving interval",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default='12355',
        help="Master port number",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default='scratch',
        help="Training mode",
        choices=["scratch", "finetune"]
    )
    parser.add_argument(
        "--custom_tokenizer",
        type=str,
        default=None,
        help="Custom tokenizer path",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="brainlessgpt",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="kenotron",
        help="Wandb entity name",
    )
    parser.add_argument(
        "--reversed_training",
        type=utils.str_to_bool,
        default=False,
        help="Reverse the training sequence",
    )
    parser.add_argument(
        "--cache_dir_train",
        type=str,
        default=None,
        help="Path to the training cache directory",
    )
    parser.add_argument(
        "--cache_dir_validation",
        type=str,
        default=None,
        help="Path to the validation cache directory",
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)

    accelerator = Accelerator()
    # accelerator.init_trackers(
    #     project_name=args.wandb_project,
    #     init_kwargs={"wandb": {"entity": args.wandb_entity}},
    #     config=args.__dict__
    # )

    device = accelerator.device
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    main(0, args, world_size)
    accelerator.end_training()