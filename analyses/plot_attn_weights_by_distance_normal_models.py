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


def load_data(batch_size=1, dataset="neuroscience_bayes_fwd_validation"):
    if dataset == "neuroscience_bayes_fwd_validation":
        cache_dir_validation_fwd = os.path.join(reference_dir, "cache/neuroscience_bayes_fwd_validation.arrow")
        dataset_fwd = datasets.Dataset.load_from_disk(cache_dir_validation_fwd)
    else:
        # Load dataset from HF
        dataset_fwd = datasets.load_dataset(
            dataset,
            cache_dir=os.path.joint(reference_dir, "cache"),
        )

    # Make sure three dataloaders produce the data in the same order
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    dataloader_fwd = DataLoader(
        dataset_fwd, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, generator=generator
    )

    return dataloader_fwd


def compute_attention_metrics_by_distance(attention_weights):
    """
    Compute attention weights and normalized ranks by distance.

    Args:
        attention_weights (np.ndarray): Attention weights of shape (seq_len, seq_len);
        which has been averaged across heads and batches.
    
    Returns:
        unique_distances (np.ndarray): Unique distances from 0 to seq_len-1.
        mean_weights (np.ndarray): Mean attention weights for each unique distance.
        mean_norm_ranks (np.ndarray): Mean normalized ranks for each unique distance.

    Steps:
        1. Process lower triangular entries (including diagonal) of attention matrix.
        2. Compute mean attention weights by distance.
        3. Compute normalized ranks for each row and average by distance.
    """
    seq_len = attention_weights.shape[0]
    distances = []
    weights = []
    norm_ranks = []
    
    # Process each row to collect weights and compute normalized ranks
    for i in range(seq_len):
        # Extract lower triangular entries (j <= i, including diagonal)
        row_weights = attention_weights[i, :i+1]  # Shape: (i+1,)
        num_elements = len(row_weights)
        
        # Collect attention weights
        for j in range(i+1):  # j <= i
            distance = abs(i - j)
            weight = abs(attention_weights[i, j])  # Magnitude only
            distances.append(distance)
            weights.append(weight)
        
        # Compute normalized ranks
        ranks = np.argsort(np.argsort(row_weights))  # Ranks from 0 to num_elements-1
        normalized_ranks = ranks / (num_elements - 1) if num_elements > 1 else np.array([0.0])
        
        # Collect normalized ranks
        for j in range(i+1):  # j <= i
            norm_rank = normalized_ranks[j]
            norm_ranks.append(norm_rank)
    
    # Compute mean weights and mean normalized ranks for each unique distance
    unique_distances = np.unique(distances)
    mean_weights = []
    mean_norm_ranks = []
    
    for dist in unique_distances:
        indices = [idx for idx, d in enumerate(distances) if d == dist]
        dist_weights = [weights[idx] for idx in indices]
        dist_norm_ranks = [norm_ranks[idx] for idx in indices]
        mean_weights.append(np.mean(dist_weights))
        mean_norm_ranks.append(np.mean(dist_norm_ranks))
    
    return unique_distances, mean_weights, mean_norm_ranks
    

def get_attention_weights_per_batch_mean_head(
        ids1,
        model1,
        attn_weights_x_batches,
    ):

    # Get the attention weights for all given some text input
    outputs1 = model1(**ids1, output_attentions=True)
    del model1, ids1
    torch.cuda.empty_cache()

    n_layers = len(attn_weights_x_batches["fwd"])
    for layer_index in range(n_layers):
        # Compute attention weights by distance for each model
        # (bsz, n_heads, seq_len, seq_len)
        attention_weights1 = outputs1.attentions[layer_index].detach().cpu().numpy()

        # Compute mean attention weights across head dimensions
        # (bsz, seq_len, seq_len)
        attention_weights1_mean_head = np.mean(attention_weights1, axis=1)

        # Collect batch-level attention weights
        attn_weights_x_batches["fwd"][layer_index]["mean_head_weights"].append(attention_weights1_mean_head)

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
        dataloader_fwd = load_data(batch_size=batch_size)
        model1, tokenizer1 = model_utils.load_model_and_tokenizer(model1_name)
        if "pythia" in model1_name:
            n_layers = model1.config.num_hidden_layers
        else: 
            n_layers = len(model1.transformer.h)
        print(f"n_layers: {n_layers}")

        for model_type in model_types:
            attn_weights_x_batches[model_type] = {}
            for layer_index in range(n_layers):
                attn_weights_x_batches[model_type][layer_index] = {}
                attn_weights_x_batches[model_type][layer_index]["mean_head_weights"] = []

        for batch_index, ids1 in enumerate(dataloader_fwd):
            print(f"batch_index: {batch_index}, ids1['input_ids'].shape={ids1['input_ids'].shape}")
            if batch_index >= max_num_batches:
                break
            
            attn_weights_x_batches = get_attention_weights_per_batch_mean_head(
                ids1,
                model1,
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

                # Get mean_weights_by_distance and mean_weights_norm_ranks by distance
                # Each `mean_weights_by_distance` and `mean_weights_norm_ranks` \in (num_unique_distances,)
                attn_weights_x_batches[model_key][layer_index]["unique_distances"], \
                attn_weights_x_batches[model_key][layer_index]["mean_weights_by_distance"], \
                attn_weights_x_batches[model_key][layer_index]["mean_weights_norm_ranks"] \
                    = compute_attention_metrics_by_distance(
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

    # Visualize attention weights
    # visualize_attention_weights(attn_weights_x_batches)
    visualize_attention_weights_norm_ranks(attn_weights_x_batches)
    # visualize_attention_weights_entropy(attn_weights_x_batches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot attention weights by distance")
    parser.add_argument("--model_size", type=str, default="small", help="Model size (small, medium, large)")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="neuroscience_bayes_fwd_validation", help="Dataset name")
    
    args = parser.parse_args()
    model_size = args.model_size
    random_seed = args.random_seed
    dataset = args.dataset
    model1_name = model_size  # To be consistent with neuro gpt2 models.
    model_size = model_size.replace("/", "--")
    print(f"model1_name: {model1_name}, model_size: {model_size}, random_seed: {random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_dir = "/home/ken/projects/backwards/model_training"
    model_types = ["fwd"]
    batch_size = 4
    max_num_batches = 16

    main()