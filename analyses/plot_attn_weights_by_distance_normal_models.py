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

from plot_attn_weights_by_distance import (
    collate_fn,
    compute_attention_metrics_by_distance,
    compute_attention_entropy,
)

def load_data(batch_size=1, dataset="neuroscience_bayes_fwd_validation"):
    cache_dir_validation_fwd = os.path.join(reference_dir, f"{dataset}.arrow")
    dataset_fwd = datasets.Dataset.load_from_disk(cache_dir_validation_fwd)

    # Make sure three dataloaders produce the data in the same order
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    dataloader_fwd = DataLoader(
        dataset_fwd, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, generator=generator
    )

    return dataloader_fwd
    

def get_attention_weights_n_entropy_per_batch_mean_head(
        ids1,
        model1,
        attn_weights_x_batches,
        batch_index,
        device
    ):
    """
    Compute attention weights and entropy for each layer, storing in preallocated GPU tensors.

    Args:
        ids1 (dict): Input IDs and attention masks for each model.
        model1: PyTorch models for forward, reverse, and permuted inputs.
        attn_weights_x_batches (dict): Dictionary with preallocated tensors for results.
        batch_index (int): Current batch index for storing results.
        device (torch.device): Device to perform computations on (GPU or CPU).

    Returns:
        attn_weights_x_batches (dict): Updated dictionary with attention weights and entropy.
    """
    # Ensure models and inputs are on the same device
    model1 = model1.to(device)
    ids1 = {k: v.to(device) for k, v in ids1.items()}

    with torch.no_grad():  # Disable gradient computation for inference
        outputs1 = model1(**ids1, output_attentions=True)
        del model1, ids1
    torch.cuda.empty_cache()

    n_layers = len(attn_weights_x_batches["fwd"])
    for layer_index in range(n_layers):
        # Attention weights are already on GPU
        attention_weights1 = outputs1.attentions[layer_index]

        # Compute mean attention weights across head dimensions
        attention_weights1_mean_head = torch.mean(attention_weights1, dim=1)  # (bsz, seq_len, seq_len)

        # Store in preallocated tensors
        attn_weights_x_batches["fwd"][layer_index]["mean_head_weights"][batch_index] = attention_weights1_mean_head

        # Compute per head entropy and average over heads and per batch
        attention_weights_entropy_mean_head1 = compute_attention_entropy(attention_weights1)

        # Store in preallocated tensors
        attn_weights_x_batches["fwd"][layer_index]["mean_head_entropy"][batch_index] = attention_weights_entropy_mean_head1
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

        # Customize plot
        ax.set_xlabel("Token Distance")
        ax.set_ylabel("Attention Weight\n(Norm Rank)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Layer {layer_index + 1}")
    
    plt.legend()
    plt.tight_layout()
    fig_fpath = f'figs/attn_weights_norm_ranks_by_distance_{model_size}_seed{model_seed}_seed{random_seed}.png'
    if "neuroscience" not in dataset:
        fig_fpath = f'figs/attn_weights_norm_ranks_by_distance_{model_size}_seed{model_seed}_seed{random_seed}_{dataset}.png'
    plt.savefig(fig_fpath)
    print(f"Saved attention weights norm ranks by distance plot to disk: {fig_fpath}")


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
        ]
        bar_heights = [float(height) for height in bar_heights]
        bar_labels = ["Fwd"]
        ax.bar(bar_x, bar_heights, color=["blue"], alpha=0.5)
        ax.set_xticks(bar_x)
        ax.set_xticklabels(bar_labels)
        ax.set_ylabel("Mean Entropy")
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f"Layer {layer_index + 1}")

    plt.tight_layout()
    fig_fpath = f'figs/attn_weights_entropy_by_distance_{model_size}_seed{model_seed}_seed{random_seed}.png'
    if "neuroscience" not in dataset:
        fig_fpath = f'figs/attn_weights_entropy_by_distance_{model_size}_seed{model_seed}_seed{random_seed}_{dataset}.png'
    plt.savefig(fig_fpath)
    print(f"Saved attention weights entropy by distance plot to disk: {fig_fpath}")


def main():
    result_fpath = f"results/attn_weights_x_batches_{model_size}_seed{model_seed}_seed{random_seed}.pkl"
    if "neuroscience" not in dataset:
        result_fpath = f"results/attn_weights_x_batches_{model_size}_seed{model_seed}_seed{random_seed}_{dataset}.pkl"

    if not os.path.exists(result_fpath):
        print("Computing attention weights by distance...")
        attn_weights_x_batches = {}
        dataloader_fwd = load_data(batch_size=batch_size, dataset=dataset)
        if "init" in model1_name:
            torch.manual_seed(model_seed)
        model1, tokenizer1 = model_utils.load_model_and_tokenizer(model1_name)
        if "pythia" in model1_name:
            n_layers = model1.config.num_hidden_layers
            # HACK:
            # Get actual seq length from `pythia_chunk{seq_len}_pile10k_fwd_train`
            seq_len = int(dataset.split("_")[1].replace("chunk", ""))
        else: 
            n_layers = len(model1.transformer.h)
            seq_len = model1.config.n_positions
        print(f"n_layers: {n_layers}, seq_len: {seq_len}")

        # Preallocate tensors for each model type and layer
        for model_type in model_types:
            attn_weights_x_batches[model_type] = {}
            for layer_index in range(n_layers):
                attn_weights_x_batches[model_type][layer_index] = {
                    # Preallocate: (max_num_batches, bsz, seq_len, seq_len)
                    "mean_head_weights": torch.zeros(
                        max_num_batches, batch_size, seq_len, seq_len, device=device
                    ),
                    # Preallocate: (max_num_batches, 1)
                    "mean_head_entropy": torch.zeros(max_num_batches, 1, device=device),
                }

        for batch_index, ids1 in enumerate(dataloader_fwd):
            print(f"batch_index: {batch_index}, ids1['input_ids'].shape={ids1['input_ids'].shape}")
            if batch_index >= max_num_batches:
                break
            
            attn_weights_x_batches = get_attention_weights_n_entropy_per_batch_mean_head(
                ids1,
                model1,
                attn_weights_x_batches,
                batch_index,
                device,
            )

        # Average the attention weights and entropy across batches
        # - Each `mean_head_weights` \in (num_batches, bsz, seq_len, seq_len)
        #   Need to average `num_batches * bsz` to get (seq_len, seq_len)
        # - Each `mean_head_entropy` \in (num_batches, 1)
        #   Need to average `num_batches` to get (1,)
        for model_key in attn_weights_x_batches:
            for layer_index in attn_weights_x_batches[model_key]:
                print(f"model_key: {model_key}, layer_index: {layer_index}")
                # Each `mean_head_weights` \in (num_batches, bsz, seq_len, seq_len)
                # Each `mean_weights` \in (seq_len, seq_len)
                attn_weights_x_batches[model_key][layer_index]["mean_weights"] = torch.mean(
                    attn_weights_x_batches[model_key][layer_index]["mean_head_weights"], 
                    dim=(0, 1),
                )

                # Get mean_weights_by_distance and mean_weights_norm_ranks by distance
                # Each `mean_weights_by_distance` and `mean_weights_norm_ranks` \in (num_unique_distances,)
                attn_weights_x_batches[model_key][layer_index]["unique_distances"], \
                attn_weights_x_batches[model_key][layer_index]["mean_weights_by_distance"], \
                attn_weights_x_batches[model_key][layer_index]["mean_weights_norm_ranks"] \
                    = compute_attention_metrics_by_distance(
                        attn_weights_x_batches[model_key][layer_index]["mean_weights"]
                )

                # Each `mean_head_entropy` \in (num_batches, 1)
                # Each `mean_weights_entropy` \in (1,)
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
        with open(result_fpath, "wb") as f:
            pickle.dump(attn_weights_x_batches, f)
        print(f"Saved attn_weights_x_batches to disk: {result_fpath}")

    else:
        with open(result_fpath, "rb") as f:
            attn_weights_x_batches = pickle.load(f)        
        print(f"Loaded attn_weights_x_batches from disk: {result_fpath}")

    # Visualize attention weights
    visualize_attention_weights_norm_ranks(attn_weights_x_batches)
    visualize_attention_weights_entropy(attn_weights_x_batches)


if __name__ == "__main__":
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Plot attention weights by distance")
    parser.add_argument("--model_size", type=str, default="small", help="Model size (small, medium, large)")
    parser.add_argument("--model_seed", type=int, default=1, help="Random seed when training model")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument(
        "--dataset", type=str, 
        default="neuroscience_bayes_fwd_validation", 
        help="Dataset name",  
        # or `gpt2_chunk1024_pile10k_fwd_train`
        # or `pythia_chunk1024_pile10k_fwd_train`
        # or `pythia_chunk2048_pile10k_fwd_train`
    )
    
    args = parser.parse_args()
    model_size = args.model_size
    model_seed = args.model_seed
    random_seed = args.random_seed
    dataset = args.dataset
    model1_name = model_size  # To be consistent with neuro gpt2 models.
    model_size = model_size.replace("/", "--")
    print(
        f"model1_name: {model1_name}, model_size: {model_size}, "
        f"model_seed: {model_seed}, random_seed: {random_seed}, "
        f"dataset: {dataset}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_dir = "/home/ken/projects/backwards/model_training/cache"
    model_types = ["fwd"]
    batch_size = 4
    max_num_batches = 16

    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")