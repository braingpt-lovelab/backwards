import os
import pickle
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})

# Between all models for train and val loss
comparison = {
    "GPT-2 (124M)": {
        "seed1": [
            "gpt2_scratch_neuro_tokenizer_bayes_fwd", 
            "gpt2_scratch_neuro_tokenizer_bayes_rev", 
            "gpt2_scratch_neuro_tokenizer_bayes_perm"
        ],
        "seed2": [
            "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed2", 
            "gpt2_scratch_neuro_tokenizer_bayes_rev_seed2", 
            "gpt2_scratch_neuro_tokenizer_bayes_perm_seed2"
        ],
        "seed3": [
            "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed3", 
            "gpt2_scratch_neuro_tokenizer_bayes_rev_seed3", 
            "gpt2_scratch_neuro_tokenizer_bayes_perm_seed3",
        ],
    },
    "GPT-2 (355M)": {
        "seed1": [
            "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd", 
            "gpt2-medium_scratch_neuro_tokenizer_bayes_rev", 
            "gpt2-medium_scratch_neuro_tokenizer_bayes_perm"
        ],
        "seed2": [
            "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed2", 
            "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed2", 
            "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed2"
        ],
        "seed3": [
            "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed3", 
            "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed3", 
            "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed3",
        ],
    },
    "GPT-2 (774M)": {
        "seed1": [
            "gpt2-large_scratch_neuro_tokenizer_bayes_fwd", 
            "gpt2-large_scratch_neuro_tokenizer_bayes_rev", 
            "gpt2-large_scratch_neuro_tokenizer_bayes_perm"
        ],
        "seed2": [
            "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed2", 
            "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed2", 
            "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed2"
        ],
        "seed3": [
            "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed3", 
            "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed3", 
            "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed3",
        ],
    },
}

def extract_needed_model_names():
    needed_model_names = []
    from utils.model_utils import model_list
    for model_family in model_list:
        needed_model_names.extend(list(model_list[model_family].keys()))
    return needed_model_names


def get_wandb_data():
    data_fpath = "results/x_models_train_val_losses.pkl"
    if not os.path.exists(data_fpath):
        needed_model_names = extract_needed_model_names()
        print(f"Needed model names: {needed_model_names}")
        print(f"Number of needed model names: {len(needed_model_names)}")
        
        # Initialize API
        api = wandb.Api(timeout=100)

        # Define project details
        entity = "kenotron"
        project = "brainlessgpt"
        runs = api.runs(f"{entity}/{project}")
        print(f"Found {len(runs)} runs in the project.")

        # List to store data
        all_data = {}

        # Iterate through runs
        for run in runs:
            run_config = run.config
            run_model_name = run_config["logfile"].split("/")[1]
            if run_model_name not in needed_model_names:
                continue
            print(f"\nModel name: {run_model_name}")

            history = run.history()
            training_ppl = history["Training PPL"].values
            validation_ppl = history["Validation PPL"].values
            
            # Remove nan due to logging bug but it's consistent
            # so ok to ignore for plotting.
            training_ppl = training_ppl[~pd.isna(training_ppl)]
            validation_ppl = validation_ppl[~pd.isna(validation_ppl)]

            all_data[run_model_name] = {
                "training_ppl": training_ppl,
                "validation_ppl": validation_ppl,
            }

        # Save to pkl
        with open("results/x_models_train_val_losses.pkl", "wb") as f:
            pickle.dump(all_data, f)


def plot_train_val_losses():
    # Load the data
    with open("results/x_models_train_val_losses.pkl", "rb") as f:
        all_data = pickle.load(f)

    # Set up the figure: 2 rows (train, val), 9 columns (3 models x 3 seeds x 3 sizes)
    fig, axes = plt.subplots(2, 9, figsize=(16, 4))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # Define model sizes and seeds
    model_sizes = ["GPT-2 (124M)", "GPT-2 (355M)", "GPT-2 (774M)"]
    seeds = ["seed1", "seed2", "seed3"]
    model_labels = ["Fwd", "Rev", "Perm"]
    colors = ['#E8B7D4', '#6AAB9C', '#5874DC']  # Colors for fwd, rev, perm
    alpha = 0.6
    lw = plt.rcParams['lines.linewidth'] ** 0.5

    # Iterate over model sizes and seeds to populate the subplots
    for size_idx, model_size in enumerate(model_sizes):
        for seed_idx, seed in enumerate(seeds):
            # Get the column index (3 columns per model size per seed)
            col_start = size_idx * 3 + seed_idx
            model_names = comparison[model_size][seed]

            # Plot training perplexity (top row)
            ax_train = axes[0, col_start]
            for model_idx, model_name in enumerate(model_names):
                if model_name in all_data:
                    train_ppl = all_data[model_name]["training_ppl"]
                    ax_train.plot(train_ppl, label=model_labels[model_idx], color=colors[model_idx], alpha=alpha, lw=lw)
            ax_train.set_title(f"{model_size}\n{seed}")
            ax_train.set_yscale("log")
            ax_train.spines['top'].set_visible(False)
            ax_train.spines['right'].set_visible(False)
            if col_start == 0:
                ax_train.set_ylabel("Training Loss")

            # Plot validation perplexity (bottom row)
            ax_val = axes[1, col_start]
            for model_idx, model_name in enumerate(model_names):
                if model_name in all_data:
                    val_ppl = all_data[model_name]["validation_ppl"]
                    ax_val.plot(val_ppl, label=model_labels[model_idx], color=colors[model_idx], alpha=alpha, lw=lw)
            if col_start == 0:
                ax_val.set_ylabel("Validation Loss")
            ax_val.set_xlabel("Steps")
            ax_val.set_yscale("log")
            ax_val.spines['top'].set_visible(False)
            ax_val.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.legend()
    plt.savefig("figs/train_val_losses_comparison.pdf")
    plt.close()


def main():
    # Get the wandb data
    get_wandb_data()

    # Plot the training and validation losses
    plot_train_val_losses()


if __name__ == "__main__":
    main()