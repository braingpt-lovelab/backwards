import os
import random
import argparse
import math
import time
import json

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import SchedulerType, get_scheduler
import wandb
import datasets
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig

import utils


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


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


def load_checkpoint(num_steps_per_epoch):
    checkpoint_dirs = [
        d for d in os.listdir(args.outputdir) \
            if os.path.isdir(os.path.join(args.outputdir, d)) \
                and d.startswith("checkpoint")
        ]

    checkpoint_dirs = sorted(
        checkpoint_dirs, 
        key=lambda x: tuple(map(int, x.split(".")[1].split("_"))), 
        reverse=True
    )

    latest_checkpoint = os.path.join(args.outputdir, checkpoint_dirs[0])
    logging(f"Resuming training from checkpoint: {latest_checkpoint}", args.logfile)
    
    # Get the epoch and step from the checkpoint file
    checkpoint_epoch = int(checkpoint_dirs[0].split(".")[1].split("_")[0])
    checkpoint_step = int(checkpoint_dirs[0].split(".")[1].split("_")[1])

    accelerator.load_state(latest_checkpoint)

    # We always want to start from the next step
    start_step = checkpoint_step + 1

    # Start from next epoch as we were at exact end of the previous epoch
    if start_step % num_steps_per_epoch == 0:
        start_epoch = checkpoint_epoch + 1
    # Start from the same epoch but at the next batch
    # (skip steps before the checkpoint)
    else:
        start_epoch = checkpoint_epoch
    
    logging(f"Resumed training from epoch {start_epoch}, step {start_step}", args.logfile)
    return start_epoch, start_step


def save_checkpoint(LLM, tokenizer, epoch, step, end_of_epoch=False):
    if not end_of_epoch:
        ckpt_dir = os.path.join(
            args.outputdir, f"checkpoint.{epoch}_{step}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        accelerator.save_state(ckpt_dir)
        logging(f"Checkpoint saved to {ckpt_dir}, AFTER epoch {epoch}, step {step}", args.logfile)
    else:
        # If end of epoch (which is not nec the exact end of an update)
        # we only save the model and tokenizer for the epoch because we will not
        # need to resume from this point but good to keep end of epoch weights
        # for evaluation
        ckpt_dir = os.path.join(
            args.outputdir, f"end_of_epoch_checkpoint.{epoch}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        LLM.module.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logging(f"End of epoch checkpoint saved to {ckpt_dir}, AFTER epoch {epoch}", args.logfile)


def main(args, world_size):
    logging(f"\nrank: {accelerator.process_index}", args.logfile)

    # Save model configuration
    with open(os.path.join(args.outputdir, 'model_config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load tokenizer
    if args.custom_tokenizer != "None":
        logging(f"Load custom tokenizer from {args.custom_tokenizer}", args.logfile)
        tokenizer = AutoTokenizer.from_pretrained(args.custom_tokenizer, cache_dir=args.cache_dir)
    else:
        logging(f"Load pretrained tokenizer", args.logfile)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)

    tokenized_dataset = {
        "train": datasets.Dataset.load_from_disk(args.cache_dir_train),
        "validation": datasets.Dataset.load_from_disk(args.cache_dir_validation),
    }

    logging(f"{tokenized_dataset}", args.logfile)
    logging(f"Loading {len(tokenized_dataset['train'])} samples for training", args.logfile)
    logging(f"Loading {len(tokenized_dataset['validation'])} samples for validation", args.logfile)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.random_seed)
    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        generator=train_generator,
    )

    valid_dataloader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    # Define model
    if args.train_mode == "scratch":
        logging(f"Train from scratch", args.logfile)
        model_config = AutoConfig.from_pretrained(args.model_path)
        # Update vocab size if using custom tokenizer
        if args.custom_tokenizer != "None":
            model_config.vocab_size = tokenizer.vocab_size
        LLM = AutoModelForCausalLM.from_config(model_config)

    elif args.train_mode == "finetune":
        logging(f"Finetune from {args.model_path}", args.logfile)
        LLM = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir=args.cache_dir)
        if args.custom_tokenizer != "None":
            raise ValueError("Bad idea to use custom tokenizer for finetuning?")
    else:
        raise ValueError("Invalid train mode")

    # Optimiser
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in LLM.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in LLM.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    # (done before prepare, ref: https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py#L173C3-L178C6)
    num_training_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.num_train_epochs
    print(f"len(train_dataloader) = {len(train_dataloader)}")
    num_warmup_steps = args.num_warmup_steps * num_training_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    LLM, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        LLM, optimizer, train_dataloader, valid_dataloader, lr_scheduler)
    
    start_epoch = 0
    start_step = 0
    best_val_loss = float('inf')
    num_steps_per_epoch = len(train_dataloader)  # per GPU
    print(f"After prepare, len(train_dataloader) = {len(train_dataloader)}")
    if args.resume_from_checkpoint:
        start_epoch, start_step = load_checkpoint(num_steps_per_epoch)
    
    logging("Start training", args.logfile)
    # Training loop
    LLM.train()
    early_stop_flag = False
    for epoch in range(start_epoch, args.num_train_epochs):
        if early_stop_flag:
            break

        start = time.time()
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            # Adhoc distruption for debugging
            # if epoch == 0 and i == 100:
            #     early_stop_flag = True
            #     break
            if epoch == 0 and i == 260:
                early_stop_flag = True
                break

            step = (epoch * num_steps_per_epoch) + i  # global step wrt per GPU

            logging(f"\nrank {accelerator.process_index} | epoch {epoch} | step {step} | batch {i}", args.logfile)
            logging(f"\n\nRunning epoch {epoch}, step {step}, batch {i}", args.logfile)
            logging(f"Sampled inputs[:2]: {batch['input_ids'][:2]}", args.logfile)
            logging(
                f"Random states for rank {accelerator.process_index}:\n"
                f"random={random.getstate()[1][:10]},\n" 
                f"np_random={np.random.get_state()[1][:10]},\n" 
                f"torch={torch.get_rng_state()[:10].tolist()},\n"
                f"torch_cuda={torch.cuda.get_rng_state(device=f'cuda:{accelerator.process_index}')[:10].tolist()}",
                args.logfile
            )

            if step < start_step:
                logging(f"Skip epoch {epoch}, step {step}, batch {i}", args.logfile)
                continue

            labels = batch["labels"]
            # Maunally remove labels to avoid internal loss compute
            # that does not use ignore_index
            batch = {
                "input_ids": batch["input_ids"], 
                "attention_mask": batch["attention_mask"]
            }
            shift_logits = LLM(**batch).logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute loss
            # - flatten logits and labels
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # - logsoftmax must ignore the BOS column
            mask = torch.zeros_like(shift_logits)
            bos_token_id = tokenizer.bos_token_id
            mask[:, bos_token_id] = float('-inf')  # Set the BOS column logits to -inf
            masked_shift_logits = shift_logits + mask
            
            # - Softmax and compute loss
            log_probs = torch.nn.functional.log_softmax(masked_shift_logits, dim=-1)
            true_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1).float()
            loss = -true_log_probs.mean()

            # Accum and backward
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            # Plot loss
            logging(f"\nrank: {accelerator.process_index}, step {step}, Loss: {loss.item()}", args.logfile)

            if (i + 1) % args.gradient_accumulation_steps == 0:
                logging(f"Graident accumulation at epoch {epoch}, step {step}, batch {i}", args.logfile)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (i + 1) % args.log_interval == 0 and accelerator.is_main_process:
                elasped_time = time.time() - start
                PPL = math.exp(loss.item() * args.gradient_accumulation_steps)
                logging(f"Epoch {epoch} | Batch {i}/{num_steps_per_epoch} | Training PPL: {PPL} | time {elasped_time}", args.logfile)
                accelerator.log({"Epoch": epoch, "Batch": i, "Training PPL": PPL, "Learning Rate": optimizer.param_groups[0]["lr"]})

            if args.save_interval > 0 \
                and (i + 1) % args.save_interval == 0 \
                and (i + 1) % args.gradient_accumulation_steps == 0:
                
                logging(f"Saving checkpoint at epoch {epoch}, step {step}, batch {i}", args.logfile)

                LLM.eval()
                with torch.no_grad():
                    val_loss = evaluate(LLM, valid_dataloader, tokenizer)
                    current_lr = optimizer.param_groups[0]["lr"]
                    torch.distributed.reduce(val_loss, 0)
                    val_loss = val_loss / world_size

                    # Save models
                    if accelerator.is_main_process:
                        val_ppl = math.exp(val_loss)
                        logging(f"Epoch {epoch} | Validation PPL: {val_ppl} | Learning rate: {current_lr}", args.logfile)
                        accelerator.log({"Epoch": epoch, "Batch": i, "Validation PPL": val_ppl, "Learning Rate": optimizer.param_groups[0]["lr"]})

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(LLM, tokenizer, epoch, step)
                LLM.train()
       
        # Evaluate again at the end of epoch
        if i == len(train_dataloader) - 1:
            LLM.eval()
            with torch.no_grad():
                val_loss = evaluate(LLM, valid_dataloader, tokenizer)
                current_lr = optimizer.param_groups[0]["lr"]
                torch.distributed.reduce(val_loss, 0)
                val_loss = val_loss / world_size

                # Save models
                if accelerator.is_main_process:
                    val_ppl = math.exp(val_loss)
                    logging(f"End of epoch {epoch} | Validation PPL: {val_ppl} | Learning rate: {current_lr}", args.logfile)
                    accelerator.log({"End of epoch": epoch, "Validation PPL": val_ppl, "Learning Rate": optimizer.param_groups[0]["lr"]})

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(LLM, tokenizer, epoch, step, end_of_epoch=True)
        LLM.train()


def evaluate(LLM, valid_dataloader, tokenizer):
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
            total_tokens += ntokens
            total_loss += loss * ntokens
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
    parser.add_argument(
        "--resume_from_checkpoint",
        type=utils.str_to_bool,
        default=True,
        help="Resume from checkpoint",
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    logging(f"{world_size}", args.logfile)

    accelerator = Accelerator(
        log_with="wandb", 
        rng_types=["torch", "cuda", "generator"],
    )

    # Handle WandB run ID for resuming, synchronized across processes
    wandb_run_id_path = os.path.join(args.outputdir, 'wandb_run_id.txt')
    
    if accelerator.is_main_process:
        if args.resume_from_checkpoint and os.path.isfile(wandb_run_id_path):
            # Explicit resumption requested and run ID file exists
            with open(wandb_run_id_path, 'r') as f:
                run_id = f.read().strip()
            resume_mode = "must"
            logging(f"Resuming WandB run with ID: {run_id}", args.logfile)
        else:
            # Fresh run: generate a new run ID, even if the file exists
            run_id = wandb.util.generate_id()
            os.makedirs(args.outputdir, exist_ok=True)
            with open(wandb_run_id_path, 'w') as f:
                f.write(run_id)
            resume_mode = "allow"
            if os.path.isfile(wandb_run_id_path) and not args.resume_from_checkpoint:
                logging(f"Starting fresh run with new ID: {run_id} (ignoring existing wandb_run_id.txt)", args.logfile)
            else:
                logging(f"Starting fresh run with new ID: {run_id}", args.logfile)
    else:
        # Non-main processes wait for the main process to create the file
        while not os.path.isfile(wandb_run_id_path):
            time.sleep(1)  # Wait until the file exists
        with open(wandb_run_id_path, 'r') as f:
            run_id = f.read().strip()
        resume_mode = "must" if args.resume_from_checkpoint else "allow"

    # Synchronize all processes to ensure they have the same run_id
    accelerator.wait_for_everyone()

    # Initialize WandB
    accelerator.init_trackers(
        project_name=args.wandb_project,
        init_kwargs={
            "wandb": {
                "entity": args.wandb_entity,
                "id": run_id,
                "resume": resume_mode,
            }
        },
        config=args.__dict__
    )    
    device = accelerator.device
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    main(args, world_size)
    accelerator.end_training()