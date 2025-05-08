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
    # Load the data
    with open("results/x_models_train_val_losses.pkl", "rb") as f:
        all_data = pickle.load(f)

    # Set up the figure with GridSpec for custom column spacing
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(2, 7, width_ratios=[1, 1, 1, 0.1, 1, 1, 1], wspace=0.5)
    axes = [[], []]
    for row in range(2):
        for col in range(6):
            if col < 3:
                axes[row].append(plt.subplot(gs[row, col]))
            else:
                axes[row].append(plt.subplot(gs[row, col + 1]))  # Skip the gap column
    axes = np.array(axes)

    # Define model sizes
    model_sizes = ["GPT-2 (124M)", "GPT-2 (355M)", "GPT-2 (774M)"]
    seeds = ["seed1", "seed2", "seed3"]
    model_labels = ["Fwd", "Bwd", "Perm"]
    colors = ['#E8B7D4', '#FF7B89', '#5874DC']  # Colors for fwd, rev, perm
    alpha = 1
    lw = plt.rcParams['lines.linewidth'] ** 0.8

    # Iterate over model sizes to populate the subplots
    for size_idx, model_size in enumerate(model_sizes):
        # Collect data across seeds for averaging
        train_ppl_data = {label: [] for label in model_labels}
        val_ppl_data = {label: [] for label in model_labels}

        for seed in seeds:
            model_names = comparison[model_size][seed]
            for model_idx, model_name in enumerate(model_names):
                if model_name in all_data:
                    train_ppl_data[model_labels[model_idx]].append(all_data[model_name]["training_ppl"])
                    val_ppl_data[model_labels[model_idx]].append(all_data[model_name]["validation_ppl"])

        # Calculate averages and standard deviations
        train_ppl_avg = {label: np.mean(data, axis=0) for label, data in train_ppl_data.items()}
        train_ppl_std = {label: np.std(data, axis=0) for label, data in train_ppl_data.items()}
        val_ppl_avg = {label: np.mean(data, axis=0) for label, data in val_ppl_data.items()}
        val_ppl_std = {label: np.std(data, axis=0) for label, data in val_ppl_data.items()}

        # Plot training perplexity (row 1, columns 1-3)
        ax_train = axes[0, size_idx]
        for model_idx, label in enumerate(model_labels):
            if train_ppl_avg[label].size > 0:
                x = range(len(train_ppl_avg[label]))
                ax_train.plot(x, train_ppl_avg[label], label=label, color=colors[model_idx], alpha=alpha, lw=lw)
                ax_train.fill_between(x, train_ppl_avg[label] - train_ppl_std[label], 
                                    train_ppl_avg[label] + train_ppl_std[label], 
                                    color=colors[model_idx], alpha=0.2)
                
        ax_train.set_title(f"{model_size}")
        ax_train.set_yscale("log")
        ax_train.spines['top'].set_visible(False)
        ax_train.spines['right'].set_visible(False)
        ax_train.set_xlim(0, len(train_ppl_avg[label]) - 1)
        ax_train.set_xticks([])
        if size_idx == 0:
            ax_train.set_ylabel("Train\nlog(perplexity)")

        # Plot validation perplexity (row 1, columns 4-6)
        ax_val = axes[0, size_idx + 3]
        for model_idx, label in enumerate(model_labels):
            if val_ppl_avg[label].size > 0:
                x = range(len(val_ppl_avg[label]))
                ax_val.plot(x, val_ppl_avg[label], label=label, color=colors[model_idx], alpha=alpha, lw=lw)
                ax_val.fill_between(x, val_ppl_avg[label] - val_ppl_std[label], 
                                   val_ppl_avg[label] + val_ppl_std[label], 
                                   color=colors[model_idx], alpha=0.2)
        ax_val.set_title(f"{model_size}")
        ax_val.set_yscale("log")
        ax_val.spines['top'].set_visible(False)
        ax_val.spines['right'].set_visible(False)
        ax_val.set_xlim(0, len(val_ppl_avg[label]) - 1)
        ax_val.set_xticks([])
        if size_idx == 0:
            ax_val.set_ylabel("Validation\nlog(perplexity)")

        # Plot training perplexity difference (Fwd - Bwd) (row 2, columns 1-3)
        ax_train_diff = axes[1, size_idx]
        if train_ppl_avg["Fwd"].size > 0 and train_ppl_avg["Bwd"].size > 0:
            train_diff = np.log(train_ppl_avg["Fwd"]) - np.log(train_ppl_avg["Bwd"])
            train_diff_std = np.sqrt(train_ppl_std["Fwd"]**2 / train_ppl_avg["Fwd"]**2 + 
                                    train_ppl_std["Bwd"]**2 / train_ppl_avg["Bwd"]**2)
            x = range(len(train_diff))
            ax_train_diff.plot(x, train_diff, color='#57D0DB', alpha=alpha, lw=lw)
            ax_train_diff.fill_between(x, train_diff - train_diff_std, train_diff + train_diff_std, 
                                     color='#57D0DB', alpha=0.2)
        ax_train_diff.set_title(f"Fwd - Bwd")
        ax_train_diff.spines['top'].set_visible(False)
        ax_train_diff.spines['right'].set_visible(False)
        ax_train_diff.plot([0, len(train_diff)], [0, 0], color='grey', lw=1, ls='--')
        ax_train_diff.set_xlim(0, len(train_diff) - 1)
        ax_train_diff.set_ylim([-0.1, 0.1])
        ax_train_diff.set_xlabel("Logging Steps")
        ax_train_diff.set_xticks([])
        if size_idx == 0:
            ax_train_diff.set_ylabel("Train\nDifference")

        # Plot validation perplexity difference (Fwd - Bwd) (row 2, columns 4-6)
        ax_val_diff = axes[1, size_idx + 3]
        if val_ppl_avg["Fwd"].size > 0 and val_ppl_avg["Bwd"].size > 0:
            val_diff = np.log(val_ppl_avg["Fwd"]) - np.log(val_ppl_avg["Bwd"])
            val_diff_std = np.sqrt(val_ppl_std["Fwd"]**2 / val_ppl_avg["Fwd"]**2 + 
                                  val_ppl_std["Bwd"]**2 / val_ppl_avg["Bwd"]**2)
            x = range(len(val_diff))
            ax_val_diff.plot(x, val_diff, color='#57D0DB', alpha=alpha, lw=lw)
            ax_val_diff.fill_between(x, val_diff - val_diff_std, val_diff + val_diff_std, 
                                    color='#57D0DB', alpha=0.2)
            
        ax_val_diff.set_title(f"Fwd - Bwd")
        ax_val_diff.set_xlabel("Logging Steps")
        ax_val_diff.set_xticks([])
        ax_val_diff.spines['top'].set_visible(False)
        ax_val_diff.spines['right'].set_visible(False)
        ax_val_diff.plot([0, len(val_diff)], [0, 0], color='grey', lw=1, ls='--')
        ax_val_diff.set_xlim(0, len(val_diff) - 1)
        ax_val_diff.set_ylim([-0.1, 0.1])
        if size_idx == 0:
            ax_val_diff.set_ylabel("Validation\nDifference")
        
    axes[0, 5].legend()
    plt.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.10)
    plt.savefig("figs/train_val_losses_comparison.pdf")
    plt.close()

def main():
    # Get the wandb data
    get_wandb_data()

    # Plot the training and validation losses
    plot_train_val_losses()


if __name__ == "__main__":
    main()