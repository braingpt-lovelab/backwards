import os
import pickle
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 14, 'font.weight': 'normal'})

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
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    # Load the data
    with open("results/x_models_train_val_losses.pkl", "rb") as f:
        all_data = pickle.load(f)

    # Set up the figure with GridSpec
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(3, 7, width_ratios=[1, 1, 1, 0.2, 1, 1, 1], wspace=0.25)
    axes = [[], [], []]
    for row in range(3):
        for col in range(6):
            if col < 3:
                axes[row].append(plt.subplot(gs[row, col]))
            else:
                axes[row].append(plt.subplot(gs[row, col + 1]))  # Skip the gap column
    axes = np.array(axes)

    # Define model sizes and labels
    model_sizes = ["GPT-2 (124M)", "GPT-2 (355M)", "GPT-2 (774M)"]
    seeds = ["seed1", "seed2", "seed3"]
    model_labels = ["Fwd", "Bwd", "Perm"]
    colors = ['#E8B7D4', '#FF7B89', '#5874DC']  # Colors for Fwd, Bwd, Perm
    alpha = 1
    lw = plt.rcParams['lines.linewidth'] ** 2

    # Iterate over models (Fwd, Bwd, Perm)
    for model_idx, model_label in enumerate(model_labels):
        # Iterate over model sizes
        for size_idx, model_size in enumerate(model_sizes):
            # Collect data across seeds for averaging
            train_ppl_data = []
            val_ppl_data = []

            for seed in seeds:
                model_names = comparison[model_size][seed]
                model_name = model_names[model_idx]
                if model_name in all_data:
                    train_ppl_data.append(all_data[model_name]["training_ppl"])
                    val_ppl_data.append(all_data[model_name]["validation_ppl"])

            # Calculate averages and standard deviations
            train_ppl_avg = np.mean(train_ppl_data, axis=0) if train_ppl_data else np.array([])
            train_ppl_std = np.std(train_ppl_data, axis=0) if train_ppl_data else np.array([])
            val_ppl_avg = np.mean(val_ppl_data, axis=0) if val_ppl_data else np.array([])
            val_ppl_std = np.std(val_ppl_data, axis=0) if val_ppl_data else np.array([])

            # Plot training perplexity (columns 1-3)
            ax_train = axes[model_idx, size_idx]
            if train_ppl_avg.size > 0:
                x = range(len(train_ppl_avg))
                ax_train.plot(x, train_ppl_avg, color=colors[model_idx], alpha=alpha, lw=lw, label=model_label)
                ax_train.fill_between(x, train_ppl_avg - train_ppl_std, 
                                    train_ppl_avg + train_ppl_std, 
                                    color=colors[model_idx], alpha=0.2)
            axes[0, size_idx].set_title(f"{model_size}")
            ax_train.set_yscale("log")
            ax_train.spines['top'].set_visible(False)
            ax_train.spines['right'].set_visible(False)
            ax_train.set_xlim(0, len(train_ppl_avg) - 1 if train_ppl_avg.size > 0 else 1)
            ax_train.set_xticks([])
            ax_train.set_ylim(1, 1e4)
            if size_idx == 0:
                ax_train.set_ylabel("Train\nperplexity")
            if size_idx != 0:
                ax_train.set_yticklabels([])

            # Plot validation perplexity (columns 4-6)
            ax_val = axes[model_idx, size_idx + 3]
            if val_ppl_avg.size > 0:
                x = range(len(val_ppl_avg))
                ax_val.plot(x, val_ppl_avg, color=colors[model_idx], alpha=alpha, lw=lw, label=model_label)
                ax_val.fill_between(x, val_ppl_avg - val_ppl_std, 
                                   val_ppl_avg + val_ppl_std, 
                                   color=colors[model_idx], alpha=0.2)
            axes[0, size_idx + 3].set_title(f"{model_size}")
            ax_val.spines['top'].set_visible(False)
            ax_val.spines['right'].set_visible(False)
            ax_val.set_xlim(0, len(val_ppl_avg) - 1 if val_ppl_avg.size > 0 else 1)
            ax_val.set_xticks([])
            ax_val.set_yscale("log")
            ax_val.set_ylim(1, 1e4)
            if size_idx == 0:
                ax_val.set_ylabel("Validation\nperplexity")
            if size_idx != 0:
                ax_val.set_yticklabels([])

            # Set xlabel for bottom row
            if model_idx == 2:
                ax_train.set_xlabel("Logging Steps")
                ax_val.set_xlabel("Logging Steps")

    axes[0, 2].legend()
    axes[0, -1].legend()
    axes[1, 2].legend()
    axes[1, -1].legend()
    axes[2, 2].legend()
    axes[2, -1].legend()
    plt.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.15)
    plt.savefig("figs/train_val_losses_comparison.pdf")
    plt.close()

def main():
    # Get the wandb data
    get_wandb_data()

    # Plot the training and validation losses
    plot_train_val_losses()


if __name__ == "__main__":
    main()