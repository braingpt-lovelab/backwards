import argparse
import datasets
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils import model_utils

plt.rcParams.update({'font.size': 10, 'font.weight': 'bold'})

def collate_fn(batch):
    input_ids = [sample["input_ids"] for sample in batch]
    attention_masks = [sample["attention_mask"] for sample in batch]
    labels = input_ids

    # Convert to tensor
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_masks.to(device), 
        "labels": labels.to(device),
    }


def load_data(batch_size=1):
    cache_dir_validation_fwd = os.path.join(reference_dir, "cache/neuroscience_bayes_fwd_validation.arrow")
    dataset_fwd = datasets.Dataset.load_from_disk(cache_dir_validation_fwd)

    cache_dir_validation_rev = os.path.join(reference_dir, "cache/neuroscience_bayes_rev_validation.arrow")
    dataset_rev = datasets.Dataset.load_from_disk(cache_dir_validation_rev)

    cache_dir_validation_perm = os.path.join(reference_dir, "cache/neuroscience_bayes_perm_validation.arrow")
    dataset_perm = datasets.Dataset.load_from_disk(cache_dir_validation_perm)

    # Make sure three dataloaders produce the data in the same order
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    dataloader_fwd = DataLoader(
        dataset_fwd, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, generator=generator
    )

    generator = torch.Generator()
    generator.manual_seed(random_seed)
    dataloader_rev = DataLoader(
        dataset_rev, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, generator=generator
    )

    generator = torch.Generator()
    generator.manual_seed(random_seed)
    dataloader_perm = DataLoader(
        dataset_perm, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, generator=generator
    )
    return dataloader_fwd, dataloader_rev, dataloader_perm


def compute_attention_by_distance(attention_weights):
    seq_len = attention_weights.shape[0]
    distances = []
    weights = []
    
    # Collect attention weights and their corresponding token distances
    # Only include lower triangular entries (i >= j) to skip masked upper triangular
    for i in range(seq_len):
        for j in range(i + 1):  # j <= i ensures we only include lower triangle + diagonal
            distance = abs(i - j)
            weight = abs(attention_weights[i, j])  # magnitude only
            distances.append(distance)
            weights.append(weight)
    
    unique_distances = np.unique(distances)
    mean_weights = []
    for dist in unique_distances:
        indices = [idx for idx, d in enumerate(distances) if d == dist]
        dist_weights = [weights[idx] for idx in indices]
        mean_weight = np.mean(dist_weights)
        mean_weights.append(mean_weight)
    
    return unique_distances, mean_weights


def compute_attention_norm_rank_by_distance(attention_weights):
    """
    Compute normalized ranks of attention weights by distance.

    Args:
        attention_weights (np.ndarray): Attention weights of shape (seq_len, seq_len);
        which has been averaged across heads and batches.
    
    Returns:
        unique_distances (np.ndarray): Unique distances from 0 to seq_len-1.
        mean_norm_ranks (np.ndarray): Mean normalized ranks for each unique distance.

    Steps:
        1. Each row of attention matrix (lower triangular + diagonal) is processed to compute normalized ranks.
        2. For each row, the ranks of the weights are computed and normalized to [0, 1].
        3. Same as `compute_attention_by_distance`, but instead of weights, we use normalized ranks.
    """
    seq_len = attention_weights.shape[0]
    distances = []
    norm_ranks = []
    
    # Process each row to compute normalized ranks for lower triangular entries
    for i in range(seq_len):
        # Extract lower triangular entries (j <= i, including diagonal)
        row_weights = attention_weights[i, :i+1]  # Shape: (i+1,)
        num_elements = len(row_weights)
        
        # Compute ranks: argsort gives indices that would sort the array
        # We want higher weights to have higher ranks
        ranks = np.argsort(np.argsort(row_weights))  # Ranks from 0 (lowest) to num_elements-1 (highest)
        
        # Normalize ranks to [0, 1]
        if num_elements > 1:
            normalized_ranks = ranks / (num_elements - 1)  # Scale to [0, 1]
        else:
            normalized_ranks = np.array([0.0])  # Single element case (i=0, j=0)
        
        # Collect distances and normalized ranks
        for j in range(i+1):  # j <= i
            distance = abs(i - j)
            norm_rank = normalized_ranks[j]
            distances.append(distance)
            norm_ranks.append(norm_rank)
    
    # Compute mean normalized rank for each unique distance
    unique_distances = np.unique(distances)
    mean_norm_ranks = []
    for dist in unique_distances:
        indices = [idx for idx, d in enumerate(distances) if d == dist]
        dist_norm_ranks = [norm_ranks[idx] for idx in indices]
        mean_norm_rank = np.mean(dist_norm_ranks)
        mean_norm_ranks.append(mean_norm_rank)
    
    return unique_distances, mean_norm_ranks


def get_attention_weights_per_batch_mean_head(
        ids1, ids2, ids3,
        model1, model2, model3,
        attn_weights_x_batches,
    ):

    # Get the attention weights for all given some text input
    outputs1 = model1(**ids1, output_attentions=True)
    del model1, ids1
    outputs2 = model2(**ids2, output_attentions=True)
    del model2, ids2
    outputs3 = model3(**ids3, output_attentions=True)
    del model3, ids3
    torch.cuda.empty_cache()

    n_layers = len(attn_weights_x_batches["fwd"])
    for layer_index in range(n_layers):
        # Compute attention weights by distance for each model
        # (bsz, n_heads, seq_len, seq_len)
        attention_weights1 = outputs1.attentions[layer_index].detach().cpu().numpy()
        attention_weights2 = outputs2.attentions[layer_index].detach().cpu().numpy()
        attention_weights3 = outputs3.attentions[layer_index].detach().cpu().numpy()

        # Compute mean attention weights across head dimensions
        # (bsz, seq_len, seq_len)
        attention_weights1_mean_head = np.mean(attention_weights1, axis=1)
        attention_weights2_mean_head = np.mean(attention_weights2, axis=1)
        attention_weights3_mean_head = np.mean(attention_weights3, axis=1)

        # Collect batch-level attention weights
        attn_weights_x_batches["fwd"][layer_index]["mean_head_weights"].append(attention_weights1_mean_head)
        attn_weights_x_batches["rev"][layer_index]["mean_head_weights"].append(attention_weights2_mean_head)
        attn_weights_x_batches["perm"][layer_index]["mean_head_weights"].append(attention_weights3_mean_head)

    return attn_weights_x_batches


def visualize_attention_weights(attn_weights_x_batches):
    n_layers = len(attn_weights_x_batches["fwd"])
    n_cols = 6
    n_rows = int(np.ceil(n_layers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for layer_index in range(n_layers):
        ax = axes[layer_index]

        # Plot mean line
        ax.plot(
            attn_weights_x_batches["fwd"][layer_index]["unique_distances"],
            attn_weights_x_batches["fwd"][layer_index]["mean_weights_by_distance"],
            label="Fwd", color="blue", alpha=0.5
        )
        ax.plot(
            attn_weights_x_batches["rev"][layer_index]["unique_distances"],
            attn_weights_x_batches["rev"][layer_index]["mean_weights_by_distance"],
            label="Rev", color="red", alpha=0.5
        )
        ax.plot(
            attn_weights_x_batches["perm"][layer_index]["unique_distances"],
            attn_weights_x_batches["perm"][layer_index]["mean_weights_by_distance"],
            label="Perm", color="cyan", alpha=0.5
        )

        # Customize plot
        ax.set_xlabel("Token Distance")
        ax.set_ylabel("Attention Weight")
        ax.set_yscale("log")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Layer {layer_index + 1}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/attn_weights_by_distance_{model_size}_seed{random_seed}.png')
    print(f"Saved attention weights by distance plot to disk: figs/attn_weights_by_distance_{model_size}_seed{random_seed}.png")


def visualize_attention_weights_norm_ranks(attn_weights_x_batches):
    n_layers = len(attn_weights_x_batches["fwd"])
    n_cols = 6
    n_rows = int(np.ceil(n_layers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for layer_index in range(n_layers):
        ax = axes[layer_index]

        # Plot mean line
        ax.plot(
            attn_weights_x_batches["fwd"][layer_index]["unique_distances"],
            attn_weights_x_batches["fwd"][layer_index]["mean_weights_norm_ranks"],
            label="Fwd", color="blue", alpha=0.5
        )
        ax.plot(
            attn_weights_x_batches["rev"][layer_index]["unique_distances"],
            attn_weights_x_batches["rev"][layer_index]["mean_weights_norm_ranks"],
            label="Rev", color="red", alpha=0.5
        )
        ax.plot(
            attn_weights_x_batches["perm"][layer_index]["unique_distances"],
            attn_weights_x_batches["perm"][layer_index]["mean_weights_norm_ranks"],
            label="Perm", color="cyan", alpha=0.5
        )

        # Customize plot
        ax.set_xlabel("Token Distance")
        ax.set_ylabel("Attention Weight\n(Norm Rank)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Layer {layer_index + 1}")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/attn_weights_norm_ranks_by_distance_{model_size}_seed{random_seed}.png')
    print(f"Saved attention weights norm ranks by distance plot to disk: figs/attn_weights_norm_ranks_by_distance_{model_size}_seed{random_seed}.png")


def main():
    if not os.path.exists(f"results/attn_weights_x_batches_{model_size}_seed{random_seed}.pkl"):
        print("Computing attention weights by distance...")
        attn_weights_x_batches = {}
        dataloader_fwd, dataloader_rev, dataloader_perm = load_data(batch_size=batch_size)
        model1, tokenizer1 = model_utils.load_model_and_tokenizer(model1_name)
        model2, tokenizer2 = model_utils.load_model_and_tokenizer(model2_name)
        model3, tokenizer3 = model_utils.load_model_and_tokenizer(model3_name)
        n_layers = len(model1.transformer.h)

        for model_type in model_types:
            attn_weights_x_batches[model_type] = {}
            for layer_index in range(n_layers):
                attn_weights_x_batches[model_type][layer_index] = {}
                attn_weights_x_batches[model_type][layer_index]["mean_head_weights"] = []

        for batch_index, (ids1, ids2, ids3) in enumerate(zip(dataloader_fwd, dataloader_rev, dataloader_perm)):
            print(f"batch_index: {batch_index}, ids1['input_ids'].shape={ids1['input_ids'].shape}")
            # print(f"ids1['input_ids'][0][1:6]: {tokenizer1.convert_ids_to_tokens(ids1['input_ids'][0][1:6])}")
            # print(f"ids2['input_ids'][0][-5:]: {tokenizer2.convert_ids_to_tokens(ids2['input_ids'][0][-5:])}")
            if batch_index >= max_num_batches:
                break
            
            attn_weights_x_batches = get_attention_weights_per_batch_mean_head(
                ids1, ids2, ids3,
                model1, model2, model3,
                attn_weights_x_batches,
            )

        # Average the attention weights across batches
        # Each `mean_head_weights` \in (num_batches, bsz, seq_len, seq_len)
        # Need to average `num_batches * bsz` to get (seq_len, seq_len)
        for model_key in attn_weights_x_batches:
            for layer_index in attn_weights_x_batches[model_key]:
                print(f"model_key: {model_key}, layer_index: {layer_index}")
                # Each `mean_head_weights` \in (num_batches, bsz, seq_len, seq_len)
                # Each `mean_weights` \in (seq_len, seq_len)
                attn_weights_x_batches[model_key][layer_index]["mean_weights"] = np.mean(
                    attn_weights_x_batches[model_key][layer_index]["mean_head_weights"], 
                    axis=(0, 1)
                )

                # Get mean_weights by distance
                # Each `mean_weights_by_distance` \in (num_unique_distances,)
                attn_weights_x_batches[model_key][layer_index]["unique_distances"], \
                    attn_weights_x_batches[model_key][layer_index]["mean_weights_by_distance"] = compute_attention_by_distance(
                        attn_weights_x_batches[model_key][layer_index]["mean_weights"]
                )
                
        # Save `attn_weights_x_batches` to disk
        with open(f"results/attn_weights_x_batches_{model_size}_seed{random_seed}.pkl", "wb") as f:
            pickle.dump(attn_weights_x_batches, f)
        print(f"Saved attn_weights_x_batches to disk: results/attn_weights_x_batches_{model_size}_seed{random_seed}.pkl")

    else:
        with open(f"results/attn_weights_x_batches_{model_size}_seed{random_seed}.pkl", "rb") as f:
            attn_weights_x_batches = pickle.load(f)
        print(f"Loaded attn_weights_x_batches from disk: results/attn_weights_x_batches_{model_size}_seed{random_seed}.pkl")

        # HACK: to incrementally get `mean_weights_norm_ranks`
        # Check if `attn_weights_x_batches[model_key][layer_index]` has key `mean_weights_norm_ranks`
        # if not, compute it
        if "mean_weights_norm_ranks" not in attn_weights_x_batches["fwd"][0]:
            print("Computing mean_weights_norm_ranks by distance...")
            for model_key in attn_weights_x_batches:
                for layer_index in attn_weights_x_batches[model_key]:
                    print(f"model_key: {model_key}, layer_index: {layer_index}")

                    # Get mean_weights_norm_ranks by distance
                    # Each `mean_weights_by_distance` \in (num_unique_distances,)
                    attn_weights_x_batches[model_key][layer_index]["unique_distances"], \
                    attn_weights_x_batches[model_key][layer_index]["mean_weights_norm_ranks"] \
                        = compute_attention_norm_rank_by_distance(
                            attn_weights_x_batches[model_key][layer_index]["mean_weights"]
                    )
            
            # Save `attn_weights_x_batches` to disk
            with open(f"results/attn_weights_x_batches_{model_size}_seed{random_seed}.pkl", "wb") as f:
                pickle.dump(attn_weights_x_batches, f)
            print(f"Saved attn_weights_x_batches to disk: results/attn_weights_x_batches_{model_size}_seed{random_seed}.pkl")

    # Visualize attention weights
    # visualize_attention_weights(attn_weights_x_batches)
    visualize_attention_weights_norm_ranks(attn_weights_x_batches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot attention weights by distance")
    parser.add_argument("--model_size", type=str, default="small", help="Model size (small, medium, large)")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed for reproducibility")
    args = parser.parse_args()
    model_size = args.model_size
    random_seed = args.random_seed

    if model_size == "small":
        model1_name = "gpt2_scratch_neuro_tokenizer_bayes_fwd"
        model2_name = "gpt2_scratch_neuro_tokenizer_bayes_rev"
        model3_name = "gpt2_scratch_neuro_tokenizer_bayes_perm"
    elif model_size == "medium":
        model1_name = "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd"
        model2_name = "gpt2-medium_scratch_neuro_tokenizer_bayes_rev"
        model3_name = "gpt2-medium_scratch_neuro_tokenizer_bayes_perm"
    elif model_size == "large":
        model1_name = "gpt2-large_scratch_neuro_tokenizer_bayes_fwd"
        model2_name = "gpt2-large_scratch_neuro_tokenizer_bayes_rev"
        model3_name = "gpt2-large_scratch_neuro_tokenizer_bayes_perm"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_dir = "/home/ken/projects/backwards/model_training"
    model_types = ["fwd", "rev", "perm"]
    batch_size = 4
    max_num_batches = 16

    main()