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
        "input_ids": input_ids,
        "attention_mask": attention_masks, 
        "labels": labels,
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


def compute_attention_norm_ranks_by_distance(attention_weights):
    """
    Compute attention weights normalized ranks by distance and mean normalized ranks by column.

    Args:
        attention_weights (torch.Tensor): Attention weights of shape (bsz, n_heads, seq_len, seq_len);
                                         Assumed to be valid probabilities, on device.

    Returns:
        unique_distances (torch.Tensor): Unique distances from 0 to seq_len-1.
        mean_norm_ranks (torch.Tensor): Mean normalized ranks for each unique distance.
        col_mean_norm_ranks (torch.Tensor): Mean normalized ranks for each column, shape (seq_len,).
    """
    device = attention_weights.device
    dtype = attention_weights.dtype
    bsz, n_heads, seq_len, _ = attention_weights.shape

    # Get lower triangular indices (including diagonal)
    i, j = torch.tril_indices(seq_len, seq_len, device=device)  # i >= j
    distances = torch.abs(i - j)  # Vectorized distance computation
    # Shape: (bsz, n_heads, num_tril_elements)
    weights = attention_weights[:, :, i, j]

    # Initialize normalized ranks
    # Shape: (bsz, n_heads, num_tril_elements)
    norm_ranks = torch.zeros_like(weights, device=device, dtype=dtype)

    # Compute normalized ranks for each row
    for row_idx in range(seq_len):
        # Extract weights for the lower triangular part of the current row
        # Shape: (bsz, n_heads, num_elements)
        row_weights = attention_weights[:, :, row_idx, :row_idx+1]
        num_elements = row_weights.shape[-1]
        
        if num_elements > 1:
            # Compute ranks: argsort(argsort) gives ranks from 0 to num_elements-1
            # Higher valued rank means higher attention weight
            ranks = torch.argsort(torch.argsort(row_weights, dim=-1), dim=-1)
            # Normalize ranks and cast to input dtype
            # Higher attention weight gets normalized rank closer to 1
            normalized_ranks = (ranks / (num_elements - 1)).to(dtype=dtype)
        else:
            normalized_ranks = torch.zeros(bsz, n_heads, 1, device=device, dtype=dtype)
        
        # Assign to correct positions using tril_indices
        mask = i == row_idx
        norm_ranks[:, :, mask] = normalized_ranks

    # Compute mean normalized ranks for each unique distance
    unique_distances = torch.unique(distances)
    mean_norm_ranks = torch.zeros(len(unique_distances), device=device, dtype=dtype)
    for dist in unique_distances:
        mask = distances == dist
        mean_norm_ranks[dist] = norm_ranks[:, :, mask].mean()

    # Compute mean normalized ranks for each column
    col_mean_norm_ranks = torch.zeros(seq_len, device=device, dtype=dtype)
    for col_idx in range(seq_len):
        mask = j == col_idx
        col_mean_norm_ranks[col_idx] = norm_ranks[:, :, mask].mean()

    return unique_distances, mean_norm_ranks, col_mean_norm_ranks


def compute_attention_entropy(attention_weights):
    """
    Compute average entropy of attention weights for each head and batch item.

    Args:
        attention_weights (torch.Tensor): Attention weights of shape (bsz, n_heads, seq_len, seq_len)
                                         Assumed to be valid probabilities, on device.

    Returns:
        mean_entropies (torch.Tensor): Mean normalized over bsz*n_heads entropies of shape (1,).
    """
    device = attention_weights.device
    bsz, n_heads, seq_len, _ = attention_weights.shape

    # Create a mask for lower triangular entries (including diagonal)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    # Expand mask to match attention_weights shape: (1, 1, seq_len, seq_len)
    mask = mask.view(1, 1, seq_len, seq_len)

    # Extract lower triangular probabilities for each row
    masked_probs = attention_weights * mask

    # Compute row entropies
    log_probs = torch.log(masked_probs + 1e-7)  # Avoid log(0)
    entropy_terms = -masked_probs * log_probs
    # Sum along the last dimension to get row entropies
    row_entropies = torch.sum(entropy_terms, dim=-1)  # Shape: (bsz, n_heads, seq_len)
    row_entropies[:, :, 0] = 0.0

    # Compute maximum possible entropies for each row
    # Uniform distribution: 1/k for k = 1, 2, ..., seq_len
    k = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32)
    uniform_prob = 1.0 / k
    max_entropies = -k * (uniform_prob * torch.log(uniform_prob + 1e-10))
    max_entropies[0] = 0.0
    max_entropies = max_entropies.view(1, 1, -1)  # Shape: (1, 1, seq_len)

    # Normalize row entropies
    normalized_entropies = torch.zeros_like(row_entropies)
    normalized_entropies[:, :, 1:] = row_entropies[:, :, 1:] / max_entropies[:, :, 1:]
    normalized_entropies[:, :, 0] = 0

    # Compute mean of normalized entropies along bsz and heads
    mean_entropies_per_row = torch.mean(normalized_entropies, dim=(0, 1))  # Shape: (seq_len,)

    # Compute mean of normalized entropies across all rows
    mean_entropies = torch.mean(mean_entropies_per_row, dim=0, keepdim=True)  # Shape: (1,)

    return mean_entropies_per_row, mean_entropies


def get_attention_weights_n_entropy_per_batch_mean_head(
        ids1, ids2, ids3,
        model1, model2, model3,
        attn_weights_x_batches,
        batch_index,
        device
    ):
    """
    Compute attention weights norm ranks and entropy for each layer, storing in preallocated GPU tensors.

    Args:
        ids1, ids2, ids3 (dict): Input IDs and attention masks for each model.
        model1, model2, model3 (torch.nn.Module): Models for each type.
        attn_weights_x_batches (dict): Preallocated tensors for storing attention weights and entropy.
        batch_index (int): Index of the current batch.
        device (torch.device): Device to run the models on.

    Returns:
        attn_weights_x_batches (dict): Updated dictionary with attention weights and entropy.
    """
    # Ensure models and inputs are on the same device
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    ids1 = {k: v.to(device) for k, v in ids1.items()}
    ids2 = {k: v.to(device) for k, v in ids2.items()}
    ids3 = {k: v.to(device) for k, v in ids3.items()}

    with torch.no_grad():  # Disable gradient computation for inference
        outputs1 = model1(**ids1, output_attentions=True)
        del model1, ids1
        outputs2 = model2(**ids2, output_attentions=True)
        del model2, ids2
        outputs3 = model3(**ids3, output_attentions=True)
        del model3, ids3
    torch.cuda.empty_cache()

    n_layers = len(attn_weights_x_batches["fwd"])
    for layer_index in range(n_layers):
        # Attention weights are already on GPU
        attention_weights1 = outputs1.attentions[layer_index]
        attention_weights2 = outputs2.attentions[layer_index]
        attention_weights3 = outputs3.attentions[layer_index]

        # Compute per head norm ranks, col norm ranks and average over heads and per batch
        unique_distances, attention_weights_norm_ranks_mean_head1, attention_weights_col_norm_ranks_mean_head1 \
            = compute_attention_norm_ranks_by_distance(attention_weights1)
        _, attention_weights_norm_ranks_mean_head2, attention_weights_col_norm_ranks_mean_head2 \
            = compute_attention_norm_ranks_by_distance(attention_weights2)
        unique_distances, attention_weights_norm_ranks_mean_head3, attention_weights_col_norm_ranks_mean_head3 \
            = compute_attention_norm_ranks_by_distance(attention_weights3)

        # Store in preallocated tensors
        attn_weights_x_batches["fwd"][layer_index]["mean_head_weights_norm_ranks"][batch_index] = attention_weights_norm_ranks_mean_head1
        attn_weights_x_batches["rev"][layer_index]["mean_head_weights_norm_ranks"][batch_index] = attention_weights_norm_ranks_mean_head2
        attn_weights_x_batches["perm"][layer_index]["mean_head_weights_norm_ranks"][batch_index] = attention_weights_norm_ranks_mean_head3
        attn_weights_x_batches["fwd"][layer_index]["mean_head_weights_col_norm_ranks"][batch_index] = attention_weights_col_norm_ranks_mean_head1
        attn_weights_x_batches["rev"][layer_index]["mean_head_weights_col_norm_ranks"][batch_index] = attention_weights_col_norm_ranks_mean_head2
        attn_weights_x_batches["perm"][layer_index]["mean_head_weights_col_norm_ranks"][batch_index] = attention_weights_col_norm_ranks_mean_head3
        attn_weights_x_batches["fwd"][layer_index]["unique_distances"] = unique_distances
        attn_weights_x_batches["rev"][layer_index]["unique_distances"] = unique_distances
        attn_weights_x_batches["perm"][layer_index]["unique_distances"] = unique_distances

        # Compute per head entropy and average over heads and per batch
        # Shape: (seq_len,) and (1,)
        attention_weights_entropy_mean_head1_per_row, attention_weights_entropy_mean_head1 \
            = compute_attention_entropy(attention_weights1)
        attention_weights_entropy_mean_head2_per_row, attention_weights_entropy_mean_head2 \
            = compute_attention_entropy(attention_weights2)
        attention_weights_entropy_mean_head3_per_row, attention_weights_entropy_mean_head3 \
            = compute_attention_entropy(attention_weights3)

        # Store in preallocated tensors
        attn_weights_x_batches["fwd"][layer_index]["mean_head_per_row_entropy"][batch_index] = attention_weights_entropy_mean_head1_per_row
        attn_weights_x_batches["rev"][layer_index]["mean_head_per_row_entropy"][batch_index] = attention_weights_entropy_mean_head2_per_row
        attn_weights_x_batches["perm"][layer_index]["mean_head_per_row_entropy"][batch_index] = attention_weights_entropy_mean_head3_per_row

        attn_weights_x_batches["fwd"][layer_index]["mean_head_entropy"][batch_index] = attention_weights_entropy_mean_head1
        attn_weights_x_batches["rev"][layer_index]["mean_head_entropy"][batch_index] = attention_weights_entropy_mean_head2
        attn_weights_x_batches["perm"][layer_index]["mean_head_entropy"][batch_index] = attention_weights_entropy_mean_head3
    return attn_weights_x_batches


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
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Layer {layer_index + 1}")

        # Only keep x and ylabels on the last row and first column
        if layer_index % n_cols != 0:
            ax.set_ylabel("")
        if layer_index < n_layers - n_cols:
            ax.set_xlabel("")

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/attn_weights_norm_ranks_by_distance_{model_size}_seed{random_seed}.pdf')
    print(f"Saved attention weights norm ranks by distance plot to disk: figs/attn_weights_norm_ranks_by_distance_{model_size}_seed{random_seed}.pdf")


def visualize_attention_weights_col_norm_ranks(attn_weights_x_batches):
    n_layers = len(attn_weights_x_batches["fwd"])
    n_cols = 6
    n_rows = int(np.ceil(n_layers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for layer_index in range(n_layers):
        ax = axes[layer_index]

        # Plot mean line
        ax.plot(
            attn_weights_x_batches["fwd"][layer_index]["mean_weights_col_norm_ranks"],
            label="Fwd", color="blue", alpha=0.5
        )
        ax.plot(
            attn_weights_x_batches["rev"][layer_index]["mean_weights_col_norm_ranks"],
            label="Rev", color="red", alpha=0.5
        )
        ax.plot(
            attn_weights_x_batches["perm"][layer_index]["mean_weights_col_norm_ranks"],
            label="Perm", color="cyan", alpha=0.5
        )

        # Customize plot
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Attention Weight\n(Norm Rank)")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Layer {layer_index + 1}")

        # Only keep x and ylabels on the last row and first column
        if layer_index % n_cols != 0:
            ax.set_ylabel("")
        if layer_index < n_layers - n_cols:
            ax.set_xlabel("")

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/attn_weights_col_norm_ranks_{model_size}_seed{random_seed}.pdf')
    print(f"Saved attention weights col norm ranks plot to disk: figs/attn_weights_col_norm_ranks_{model_size}_seed{random_seed}.pdf")


def visualize_attention_weights_entropy_per_row(attn_weights_x_batches):
    n_layers = len(attn_weights_x_batches["fwd"])
    n_cols = 6
    n_rows = int(np.ceil(n_layers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for layer_index in range(n_layers):
        ax = axes[layer_index]

        # Plot each model's mean entropy per row as curve
        ax.plot(
            attn_weights_x_batches["fwd"][layer_index]["mean_weights_per_row_entropy"],
            label="Fwd", color="blue", alpha=0.5
        )
        ax.plot(
            attn_weights_x_batches["rev"][layer_index]["mean_weights_per_row_entropy"],
            label="Rev", color="red", alpha=0.5
        )
        ax.plot(
            attn_weights_x_batches["perm"][layer_index]["mean_weights_per_row_entropy"],
            label="Perm", color="cyan", alpha=0.5
        )

        # Customize plot
        ax.set_xlabel("Context Size")
        ax.set_ylabel("Mean Entropy")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Layer {layer_index + 1}")

        # Only keep x and ylabels on the last row and first column
        if layer_index % n_cols != 0:
            ax.set_ylabel("")
        if layer_index < n_layers - n_cols:
            ax.set_xlabel("")

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/attn_weights_entropy_per_row_{model_size}_seed{random_seed}.pdf')
    print(f"Saved attention weights entropy per row plot to disk: figs/attn_weights_entropy_per_row_{model_size}_seed{random_seed}.pdf")


def visualize_attention_weights_entropy(attn_weights_x_batches):
    n_layers = len(attn_weights_x_batches["fwd"])
    n_cols = 6
    n_rows = int(np.ceil(n_layers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for layer_index in range(n_layers):
        ax = axes[layer_index]

        # Plot each model's mean entropy as barplot
        bar_x = np.arange(len(model_types))
        bar_heights = [
            attn_weights_x_batches["fwd"][layer_index]["mean_weights_entropy"],
            attn_weights_x_batches["rev"][layer_index]["mean_weights_entropy"],
            attn_weights_x_batches["perm"][layer_index]["mean_weights_entropy"]
        ]
        bar_heights = [float(height) for height in bar_heights]
        bar_labels = ["Fwd", "Rev", "Perm"]
        ax.bar(bar_x, bar_heights, color=["blue", "red", "cyan"], alpha=0.5)
        ax.set_xticks(bar_x)
        ax.set_xticklabels(bar_labels)
        ax.set_ylabel("Mean Entropy")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Layer {layer_index + 1}")

        # Only keep x and ylabels on the last row and first column
        if layer_index % n_cols != 0:
            ax.set_ylabel("")
        if layer_index < n_layers - n_cols:
            ax.set_xlabel("")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'figs/attn_weights_entropy_{model_size}_seed{random_seed}.pdf')
    print(f"Saved attention weights entropy plot to disk: figs/attn_weights_entropy_{model_size}_seed{random_seed}.pdf")


def main():
    if not os.path.exists(f"results/attn_weights_x_batches_{model_size}_seed{random_seed}.pkl"):
        print("Computing attention weights by distance...")
        attn_weights_x_batches = {}
        dataloader_fwd, dataloader_rev, dataloader_perm = load_data(batch_size=batch_size)
        model1, tokenizer1 = model_utils.load_model_and_tokenizer(model1_name)
        model2, tokenizer2 = model_utils.load_model_and_tokenizer(model2_name)
        model3, tokenizer3 = model_utils.load_model_and_tokenizer(model3_name)
        n_layers = len(model1.transformer.h)
        seq_len = model1.config.n_positions
        # diff by 0, 1, ..., seq_len-1
        num_unique_distances = seq_len

        # Preallocate tensors for each model type and layer
        for model_type in model_types:
            attn_weights_x_batches[model_type] = {}
            for layer_index in range(n_layers):
                attn_weights_x_batches[model_type][layer_index] = {
                    # Preallocate: (max_num_batches, num_unique_distances)
                    "mean_head_weights_norm_ranks": torch.zeros(
                        max_num_batches, seq_len, device=device
                    ),
                    # Preallocate: (max_num_batches, seq_len)
                    "mean_head_weights_col_norm_ranks": torch.zeros(
                        max_num_batches, seq_len, device=device
                    ),
                    # Preallocate: (num_unique_distances,)
                    "unique_distances": torch.zeros(
                        num_unique_distances, device=device
                    ),
                    # Preallocate: (max_num_batches, seq_len)
                    "mean_head_per_row_entropy": torch.zeros(
                        max_num_batches, seq_len, device=device
                    ),
                    # Preallocate: (max_num_batches, 1)
                    "mean_head_entropy": torch.zeros(max_num_batches, 1, device=device),
                }

        for batch_index, (ids1, ids2, ids3) in enumerate(zip(dataloader_fwd, dataloader_rev, dataloader_perm)):
            print(f"batch_index: {batch_index}, ids1['input_ids'].shape={ids1['input_ids'].shape}")
            if batch_index >= max_num_batches:
                break
            
            attn_weights_x_batches = get_attention_weights_n_entropy_per_batch_mean_head(
                ids1, ids2, ids3,
                model1, model2, model3,
                attn_weights_x_batches,
                batch_index,
                device,
            )

        # Average the attention weights norm ranks and entropy across batches
        # - Each `mean_head_weights_norm_ranks` \in (num_batches, num_unique_distances)
        #   Need to average `num_batches` to get (num_unique_distances,)
        # - Each `mean_head_weights_col_norm_ranks` \in (num_batches, seq_len)
        #   Need to average `num_batches` to get (seq_len,)
        # - Each `mean_head_per_row_entropy` \in (num_batches, seq_len)
        #   Need to average `num_batches` to get (seq_len,)
        # - Each `mean_head_entropy` \in (num_batches, 1)
        #   Need to average `num_batches` to get (1,)
        for model_key in attn_weights_x_batches:
            for layer_index in attn_weights_x_batches[model_key]:
                print(f"model_key: {model_key}, layer_index: {layer_index}")
                attn_weights_x_batches[model_key][layer_index]["mean_weights_norm_ranks"] = torch.mean(
                    attn_weights_x_batches[model_key][layer_index]["mean_head_weights_norm_ranks"],
                    dim=0,
                )

                attn_weights_x_batches[model_key][layer_index]["mean_weights_col_norm_ranks"] = torch.mean(
                    attn_weights_x_batches[model_key][layer_index]["mean_head_weights_col_norm_ranks"],
                    dim=0,
                )

                attn_weights_x_batches[model_key][layer_index]['mean_weights_per_row_entropy'] = torch.mean(
                    attn_weights_x_batches[model_key][layer_index]["mean_head_per_row_entropy"],
                    dim=0,
                )

                attn_weights_x_batches[model_key][layer_index]["mean_weights_entropy"] = torch.mean(
                    attn_weights_x_batches[model_key][layer_index]["mean_head_entropy"], 
                    dim=0,
                )

        # Move all tensors to CPU before saving
        for model_key in attn_weights_x_batches:
            for layer_index in attn_weights_x_batches[model_key]:
                for key in attn_weights_x_batches[model_key][layer_index]:
                    attn_weights_x_batches[model_key][layer_index][key] = (
                        attn_weights_x_batches[model_key][layer_index][key].cpu()
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
    visualize_attention_weights_norm_ranks(attn_weights_x_batches)
    visualize_attention_weights_col_norm_ranks(attn_weights_x_batches)
    visualize_attention_weights_entropy_per_row(attn_weights_x_batches)
    visualize_attention_weights_entropy(attn_weights_x_batches)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import time
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Plot attention weights by distance")
    parser.add_argument("--model_size", type=str, default="small", help="Model size (small, medium, large)")
    parser.add_argument("--model_seed", type=int, default=1, help="Random seed when training model")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed for sampling")
    args = parser.parse_args()
    model_size = args.model_size
    model_seed = args.model_seed
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
    
    # NOTE: for now we use the same fwd and rev models for convenience but
    # varying perm model seeds. But ideally, we should have the same seed for
    # fwd, rev, and perm models.
    if int(model_seed) > 1:
        model3_name += f"_seed{model_seed}"
        model_size += f"_seed{model_seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_dir = "/home/ken/projects/backwards/model_training"
    model_types = ["fwd", "rev", "perm"]
    batch_size = 4
    max_num_batches = 16
    if not os.path.exists("results"):
        os.makedirs("results")

    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")