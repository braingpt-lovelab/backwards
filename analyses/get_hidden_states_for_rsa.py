import argparse
import os
import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import model_utils


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


def load_data(batch_size, model_name):
    if "fwd" in model_name.lower():
        cache_dir_validation = os.path.join(reference_dir, "cache/neuroscience_bayes_fwd_validation.arrow")
    elif "rev" in model_name.lower():
        cache_dir_validation = os.path.join(reference_dir, "cache/neuroscience_bayes_rev_validation.arrow")
    elif "perm" in model_name.lower():
        cache_dir_validation = os.path.join(reference_dir, "cache/neuroscience_bayes_perm_validation.arrow")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    dataset = datasets.Dataset.load_from_disk(cache_dir_validation)

    # Make sure three dataloaders produce the data in the same order
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=collate_fn, 
        shuffle=True, 
        generator=generator,
    )
    return dataloader


def main():
    for model_name in model_names:
        model, _ = model_utils.load_model_and_tokenizer(model_name)
        dataloader = load_data(
            batch_size=batch_size, model_name=model_name
        )
        for batch_index, ids in enumerate(dataloader):
            if batch_index >= max_num_batches:
                break

            # Get the model's hidden states
            # shape: (bsz, num_layers, seq_len, hidden_size)
            ids = {k: v.to(device) for k, v in ids.items()}
            outputs = model(
                **ids, 
                output_attentions=False,
                output_hidden_states=True,
            )
            all_hidden_states_except_emb = outputs.hidden_states[1:]

            # Save hidden states per layer, per batch
            for layer_index, hidden_states in enumerate(all_hidden_states_except_emb):
                # Save the hidden states
                fdir = os.path.join(results_dir, model_name, "hidden_states")
                if not os.path.exists(fdir):
                    os.makedirs(fdir)
                fpath = os.path.join(
                    fdir,
                    f"hidden_states_layer{layer_index}_batch{batch_index}_seed{random_seed}.pt"
                )
                torch.save(hidden_states, fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed for sampling")
    args = parser.parse_args()

    random_seed = args.random_seed
    batch_size = 4
    max_num_batches = 16
    if not os.path.exists("model_results"):
        os.makedirs("model_results")
    results_dir = "model_results"
    reference_dir = "/home/ken/projects/backwards/model_training"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = []
    model_list = model_utils.model_list
    for model_family in model_utils.model_list:
        model_names.extend(list(model_list[model_family].keys()))

    print("model_names:", model_names)
    main()

