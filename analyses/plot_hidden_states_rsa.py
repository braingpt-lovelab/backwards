import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
import transformers
import scipy.stats as stats

comparison = {
    "seed1": {
        "GPT-2 (124M)": [
            "gpt2_scratch_neuro_tokenizer_bayes_fwd",
            "gpt2_scratch_neuro_tokenizer_bayes_rev",
            "gpt2_scratch_neuro_tokenizer_bayes_perm"
        ],
        "GPT-2 (355M)": [
            "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_rev",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_perm"
        ],
        "GPT-2 (774M)": [
            "gpt2-large_scratch_neuro_tokenizer_bayes_fwd",
            "gpt2-large_scratch_neuro_tokenizer_bayes_rev",
            "gpt2-large_scratch_neuro_tokenizer_bayes_perm"
        ]
    },
    "seed2": {
        "GPT-2 (124M)": [
            "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed2",
            "gpt2_scratch_neuro_tokenizer_bayes_rev_seed2",
            "gpt2_scratch_neuro_tokenizer_bayes_perm_seed2"
        ],
        "GPT-2 (355M)": [
            "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed2",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed2",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed2"
        ],
        "GPT-2 (774M)": [
            "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed2",
            "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed2",
            "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed2"
        ]
    },
    "seed3": {
        "GPT-2 (124M)": [
            "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed3",
            "gpt2_scratch_neuro_tokenizer_bayes_rev_seed3",
            "gpt2_scratch_neuro_tokenizer_bayes_perm_seed3"
        ],
        "GPT-2 (355M)": [
            "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed3",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed3",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed3"
        ],
        "GPT-2 (774M)": [
            "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed3",
            "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed3",
            "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed3"
        ]
    }
}

def remap_permuted_indices(seq_len, perm_seed):
    """
    Given `seq_len`, first get 
        np.random.seed(random_seed)
        permuted_indices = np.random.permutation(range(seq_len-1)), # seq_len-1 due to BOS
    
    and create remapping_indices for which, if an array previously permuted
    by `permuted_indices` will be remapped to the original order.

    E.g., 
        array = [a, b, c]
        permuted_indices = [2, 0, 1]
        permuted_array = [c, a, b]
        permuted_array[remapping_indices] = array
        so remapping_indices = [1, 2, 0]
    """
    np.random.seed(perm_seed)
    permuted_indices = np.random.permutation(range(seq_len-1))  # seq_len-1 due to BOS
    remapping_indices = np.argsort(permuted_indices) + 1
    # BOS token at 0 never changes
    remapping_indices = np.concatenate(([0], remapping_indices))
    # Convert to tensor
    remapping_indices = torch.tensor(remapping_indices, dtype=torch.long)
    return remapping_indices


def reorder_hidden_states(hidden_states1, hidden_states2, model1_name, model2_name):
    """
    hidden_states: shape (bsz, seq_len, hidden_size)

    if "fwd" in model1_name and "rev" in model2_name:
        hidden_states1 stays the same, 
        hidden_states2 is flip along the seq_len dim, but only flip those [1:] rows,

    elif "fwd" in model1_name and "perm" in model2_name:
        hidden_states1 stays the same, 
        hidden_states2 is reordered along the seq_len dim, but only those [1:] rows,

        Reorder by:
            Given `seq_len`, first get 
                np.random.seed(random_seed)
                permuted_indices = np.random.permutation(range(seq_len))
            we create remapping_indices for which, if an array previously permuted
            by `permuted_indices` will be remapped to the original order.
    
    elif "rev" in model1_name and "perm" in model2_name:
        hidden_states1 is flip along the seq_len dim, but only flip those [1:] rows,
        hidden_states2 is reordered along the seq_len dim, but only those [1:] rows,
        
        Reorder by:
            Given `seq_len`, first get 
                np.random.seed(random_seed)
                permuted_indices = np.random.permutation(range(seq_len))
            we create remapping_indices for which, if an array previously permuted
            by `permuted_indices` will be remapped to the original order.
    """
    if "fwd" in model1_name and "rev" in model2_name:
        reordered_matrix1 = hidden_states1
        
        # Perform flips in-place on hidden_states2
        reordered_matrix2 = hidden_states2

        # Flip along seq_len dim (rows), excluding first row (it's always BOS)
        reordered_matrix2[:, 1:, :] = torch.flip(reordered_matrix2[:, 1:, :], dims=[1])
    
    elif "fwd" in model1_name and "perm" in model2_name:
        reordered_matrix1 = hidden_states1
        
        # Perform remapping in-place on hidden_states2
        reordered_matrix2 = hidden_states2

        # Use the same seed for previous permutation
        if "seed" in model2_name:
            # Extract the seed from the model name
            seed = int(model2_name.split("_seed")[-1])
        else:
            seed = 1

        # Reorder along seq_len dim (rows)
        seq_len = hidden_states1.shape[1]
        remapping_indices = remap_permuted_indices(seq_len, seed)
        reordered_matrix2[:, :, :] = reordered_matrix2[:, :, :][:, remapping_indices, :]
    
    elif "rev" in model1_name and "perm" in model2_name:
        # Perform flips in-place on hidden_states1
        reordered_matrix1 = hidden_states1
        # Flip along seq_len dim (rows), excluding first row (it's always BOS)
        reordered_matrix1[:, 1:, :] = torch.flip(reordered_matrix1[:, 1:, :], dims=[1])

        # Perform remapping in-place on hidden_states2
        reordered_matrix2 = hidden_states2
        # Use the same seed for previous permutation
        if "seed" in model2_name:
            # Extract the seed from the model name
            seed = int(model2_name.split("_seed")[-1])
        else:
            seed = 1

        # Reorder along seq_len dim (rows)
        seq_len = hidden_states1.shape[1]
        remapping_indices = remap_permuted_indices(seq_len, seed)
        reordered_matrix2[:, :, :] = reordered_matrix2[:, :, :][:, remapping_indices, :]
    
    else:
        raise ValueError("Invalid model names for comparison.")
    return reordered_matrix1, reordered_matrix2


def rdm(model_hidden_states_per_batch, rdm_metric="cosine"):
    """
    Compute the RDM for each model's hidden states from a batch.

    Args:
        model_hidden_states_per_batch (torch.Tensor): Hidden states of a model, shape (bsz, seq_len, hidden_size)
        rdm_metric (str): Metric to use for RDM. Default is "cosine".
    
    Notes: parallel processing for all bsz

    Returns:
        model_RDM_per_batch (torch.Tensor): RDM of the model, shape (bsz, seq_len, seq_len)
    """
    dot_product = torch.einsum(
        "bld,bds->bls",
        model_hidden_states_per_batch, 
        model_hidden_states_per_batch.transpose(1, 2)
    )

    if rdm_metric == "cosine":
        # Compute norms: sqrt of diagonal elements
        # Shape: (bsz, seq_len)
        norms = torch.sqrt(torch.diagonal(dot_product, dim1=1, dim2=2).clamp(min=0))

        # Compute cosine similarity: (x_i . x_j) / (||x_i|| ||x_j||)
        # Shape: (bsz, seq_len, seq_len)
        norms_product = norms.unsqueeze(2) * norms.unsqueeze(1) 
        # Avoid division by zero 
        cosine_sim = dot_product / norms_product.clamp(min=1e-8)

        # Compute cosine distance
        cosine_dist = 1 - cosine_sim  # Shape: (bsz, seq_len, seq_len)

        model_RDM_per_batch = cosine_dist
    
    else:
        raise ValueError("Invalid RDM metric. Only 'cosine' is supported.")

    return model_RDM_per_batch


def rsa(model1_RDM, model2_RDM, rsa_metric="spearman"):
    """
    Compute the RSA between two RDMs from a batch, using RDMs'
    lower triangular matrices including the diagonal.

    Args:
        model1_RDM (torch.Tensor): RDM of model 1, shape (bsz, seq_len, seq_len)
        model2_RDM (torch.Tensor): RDM of model 2, shape (bsz, seq_len, seq_len)
        rsa_metric (str): Metric to use for RSA. Default is "spearman".
    
    Notes: parallel processing for all bsz

    Returns:
        rsa_scores (torch.Tensor): RSA scores, shape (bsz,)
    """
    # Get the lower triangular part of the RDMs
    model1_RDM_lower = torch.tril(model1_RDM)
    model2_RDM_lower = torch.tril(model2_RDM)

    # Compute the number of lower triangular elements (including diagonal)
    bsz, seq_len, _ = model1_RDM_lower.shape
    num_lower_tri_elements = (seq_len * (seq_len + 1)) // 2

    # Pre-allocate tensors for flattened lower triangular elements
    model1_flat = torch.zeros(bsz, num_lower_tri_elements)
    model2_flat = torch.zeros(bsz, num_lower_tri_elements)

    # Flatten the lower triangular part of each batch
    for i in range(bsz):
        # Get lower triangular elements (including diagonal)
        tril_mask = torch.tril(torch.ones_like(model1_RDM_lower[i])).bool()
        model1_flat[i] = model1_RDM_lower[i][tril_mask]
        model2_flat[i] = model2_RDM_lower[i][tril_mask]

    # Compute the RSA scores
    if rsa_metric == "cosine":
        rsa_scores = F.cosine_similarity(model1_flat, model2_flat, dim=1)
    elif rsa_metric == "spearman":
        rsa_scores = torch.zeros(bsz)
        for i in range(bsz):
            # Compute Spearman correlation
            spearman_corr, _ = stats.spearmanr(model1_flat[i].cpu().numpy(), model2_flat[i].cpu().numpy())
            rsa_scores[i] = spearman_corr
    elif rsa_metric == "euclidean":
        # Compute Euclidean distance
        rsa_scores = torch.sqrt(torch.sum((model1_flat - model2_flat) ** 2, dim=1))

    return rsa_scores


def main():
    plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})
    lw = plt.rcParams['lines.linewidth'] ** 2
    
    # Directory to save/load RSA results
    os.makedirs("results", exist_ok=True)
    
    with torch.no_grad():
        for seed in comparison.keys():
            # Fig is created at seed level.
            # Each column is a model family, each subplot contains all model comparisons
            fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)
            legend_handles = []
            legend_labels = []
            axes = axes.flatten()
            ax_idx = 0
            for model_family in comparison[seed].keys():
                ax = axes[ax_idx]
                for i in range(len(comparison[seed][model_family])):
                    for j in range(i + 1, len(comparison[seed][model_family])):
                        model1_name = comparison[seed][model_family][i]
                        model2_name = comparison[seed][model_family][j]
                        model1_attn_dir = f"model_results/{model1_name}/hidden_states"
                        model2_attn_dir = f"model_results/{model2_name}/hidden_states"
                        print(f"Comparing {model1_name} and {model2_name}...")

                        # Get num_layers and hidden_size from model config 
                        # (same for model1 and model2)
                        if "medium" in model1_name:
                            gpt_config = transformers.GPT2Config.from_pretrained("gpt2-medium")
                        elif "large" in model1_name:
                            gpt_config = transformers.GPT2Config.from_pretrained("gpt2-large")
                        else:
                            gpt_config = transformers.GPT2Config.from_pretrained("gpt2")

                        num_layers = gpt_config.num_hidden_layers
                        hidden_size = gpt_config.hidden_size
                        seq_len = gpt_config.n_positions
                        print(f"num_layers: {num_layers}, hidden_size: {hidden_size}, seq_len: {seq_len}")

                        # Define file path for saving/loading RSA scores
                        rsa_result_file = (
                            f"results/rsa_{model1_name}_vs_{model2_name}_"
                            f"{rdm_metric}_{rsa_metric}_seed{seed}.pt"
                        )
                        
                        # Check if RSA results already exist
                        if os.path.exists(rsa_result_file):
                            print(f"Loading existing RSA results from {rsa_result_file}")
                            rsa_data = torch.load(rsa_result_file)
                            rsa_scores_x_layers_rsa_mean = rsa_data['mean']
                            rsa_scores_x_layers_rsa_std = rsa_data['std']
                        else:
                            # Compute RSA scores
                            rsa_scores_x_layers_rsa_mean = []
                            rsa_scores_x_layers_rsa_std = []
                            for layer_idx in range(num_layers):
                                rsa_scores_per_layer_x_batches = torch.empty((max_num_batches, batch_size))
                                for batch_idx in range(max_num_batches):
                                    # shape: (bsz, seq_len, hidden_size)
                                    model1_hidden_states_per_batch = torch.load(
                                        f"{model1_attn_dir}/hidden_states_layer{layer_idx}_batch{batch_idx}_seed{random_seed}.pt",
                                    )
                                    model2_hidden_states_per_batch = torch.load(
                                        f"{model2_attn_dir}/hidden_states_layer{layer_idx}_batch{batch_idx}_seed{random_seed}.pt",
                                    )

                                    # Reorder rows of hidden states to be fwd order.
                                    model1_hidden_states_per_batch, model2_hidden_states_per_batch \
                                        = reorder_hidden_states(
                                            model1_hidden_states_per_batch,
                                            model2_hidden_states_per_batch,
                                            model1_name,
                                            model2_name
                                    )
                                    
                                    # Compute RDM
                                    # shape: (bsz, seq_len, seq_len)
                                    model1_RDM_per_batch = rdm(
                                        model1_hidden_states_per_batch, 
                                        rdm_metric=rdm_metric
                                    )
                                    model2_RDM_per_batch = rdm(
                                        model2_hidden_states_per_batch, 
                                        rdm_metric=rdm_metric
                                    )

                                    # RSA
                                    # shape: (bsz,)
                                    rsa_scores_per_batch = rsa(
                                        model1_RDM_per_batch, 
                                        model2_RDM_per_batch,
                                        rsa_metric=rsa_metric
                                    )

                                    # collect for this batch
                                    rsa_scores_per_layer_x_batches[batch_idx] = rsa_scores_per_batch
                                
                                # Compute mean and std for this layer (
                                # i.e., reducing over both dims: max_num_batches and batch_size
                                # )
                                rsa_scores_x_layers_rsa_mean.append(rsa_scores_per_layer_x_batches.mean().item())
                                rsa_scores_x_layers_rsa_std.append(rsa_scores_per_layer_x_batches.std().item())
                                print(
                                    f"Layer {layer_idx}: mean = {rsa_scores_x_layers_rsa_mean[-1]}, "
                                    f"std = {rsa_scores_x_layers_rsa_std[-1]}"
                                )
                            
                            # Save RSA results
                            torch.save(
                                {
                                    'mean': rsa_scores_x_layers_rsa_mean,
                                    'std': rsa_scores_x_layers_rsa_std
                                },
                                rsa_result_file
                            )
                            print(f"Saved RSA results to {rsa_result_file}")

                        # Plot for all layers at once
                        if "fwd" in model1_name and "rev" in model2_name:
                            label = "Fwd vs Bwd"
                            color = "#E8B7D4"
                            ls = "-"
                        elif "fwd" in model1_name and "perm" in model2_name:
                            label = "Fwd vs Perm"
                            color = "#FF7B89"
                            ls = "--"
                        elif "rev" in model1_name and "perm" in model2_name:
                            label = "Bwd vs Perm"
                            color = "#5874DC"
                            ls = ":"
                        else:
                            raise ValueError("Invalid model names for comparison.")
                        
                        line, = ax.plot(
                            range(num_layers), rsa_scores_x_layers_rsa_mean,
                            label=label,
                            color=color,
                            linewidth=lw,
                            linestyle=ls,
                        )
                        if ax_idx == 0:
                            legend_handles.append(line)
                            legend_labels.append(label)

                        ax.fill_between(
                            range(num_layers), 
                            np.array(rsa_scores_x_layers_rsa_mean) - np.array(rsa_scores_x_layers_rsa_std),
                            np.array(rsa_scores_x_layers_rsa_mean) + np.array(rsa_scores_x_layers_rsa_std),
                            alpha=0.2,
                            color=color,
                        )

                if ax_idx == 0:
                    ax.set_ylabel(f"RSA ({rsa_metric})")
                ax.set_title(f"{model_family}")
                ax.set_xlabel("Layer")
                ax.set_ylim(-0.1, 1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True)
                ax_idx += 1
            
            fig.legend(
                legend_handles,
                legend_labels,
                loc='lower center',
                ncol=3,
                bbox_to_anchor=(0.5, 0.005),
                frameon=True
            )
        
            plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.3, wspace=0.1)
            plt.savefig(f"figs/rsa_results_{rdm_metric}_{rsa_metric}_{seed}.pdf")


if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed for text sampling")
    parser.add_argument("--rdm_metric", type=str, default="cosine", help="Metric to use for RDM")
    parser.add_argument("--rsa_metric", type=str, default="spearman", help="Metric to use for RSA")
    args = parser.parse_args()
    random_seed = args.random_seed
    rdm_metric = args.rdm_metric
    rsa_metric = args.rsa_metric

    batch_size = 4
    max_num_batches = 16

    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Total time taken: {duration/60:.2f} minutes")

