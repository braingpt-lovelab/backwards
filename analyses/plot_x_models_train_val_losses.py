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

    # Define model sizes and other parameters
    model_sizes = ["GPT-2 (124M)", "GPT-2 (355M)", "GPT-2 (774M)"]
    seeds = ["seed1", "seed2", "seed3"]
    model_labels = ["Fwd", "Bwd", "Perm"]
    colors = ['#E8B7D4', '#FF7B89', '#5874DC']  # Colors for Fwd, Bwd, Perm
    alpha = 1
    lw = plt.rcParams['lines.linewidth'] ** 2

    # Function to create a figure for either train or validation
    def create_figure(data_type, filename):
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(3, 3, wspace=0.35, hspace=0.15)
        axes = np.array([[plt.subplot(gs[row, col]) for col in range(3)] for row in range(3)])

        for size_idx, model_size in enumerate(model_sizes):
            # Collect data across seeds for averaging
            ppl_data = {label: [] for label in model_labels}
            for seed in seeds:
                model_names = comparison[model_size][seed]
                for model_idx, model_name in enumerate(model_names):
                    if model_name in all_data:
                        ppl_data[model_labels[model_idx]].append(
                            all_data[model_name][f"{data_type}_ppl"]
                        )

            # Calculate averages and standard deviations
            ppl_avg = {label: np.mean(data, axis=0) for label, data in ppl_data.items()}
            ppl_std = {label: np.std(data, axis=0) for label, data in ppl_data.items()}

            # Plot Fwd vs Perm (row 1)
            ax_fwd_perm = axes[0, size_idx]
            for label, color in zip(["Fwd", "Perm"], [colors[0], colors[2]]):
                if ppl_avg[label].size > 0:
                    x = range(len(ppl_avg[label]))
                    ax_fwd_perm.plot(x, ppl_avg[label], label=label, color=color, alpha=alpha, lw=lw)
                    ax_fwd_perm.fill_between(
                        x, ppl_avg[label] - ppl_std[label], ppl_avg[label] + ppl_std[label],
                        color=color, alpha=0.2
                    )
            ax_fwd_perm.set_title(f"{model_size}")
            ax_fwd_perm.set_yscale("log")
            ax_fwd_perm.spines['top'].set_visible(False)
            ax_fwd_perm.spines['right'].set_visible(False)
            ax_fwd_perm.set_xlim(0, len(ppl_avg["Fwd"]) - 1)
            ax_fwd_perm.set_xticks([])
            ax_fwd_perm.set_ylim([10, 1e4])
            if size_idx == 0:
                ax_fwd_perm.set_ylabel("Fwd vs Perm\nln(perplexity)")

            # Plot Bwd vs Perm (row 2)
            ax_bwd_perm = axes[1, size_idx]
            for label, color in zip(["Bwd", "Perm"], [colors[1], colors[2]]):
                if ppl_avg[label].size > 0:
                    x = range(len(ppl_avg[label]))
                    ax_bwd_perm.plot(x, ppl_avg[label], label=label, color=color, alpha=alpha, lw=lw)
                    ax_bwd_perm.fill_between(
                        x, ppl_avg[label] - ppl_std[label], ppl_avg[label] + ppl_std[label],
                        color=color, alpha=0.2
                    )
            ax_bwd_perm.set_yscale("log")
            ax_bwd_perm.spines['top'].set_visible(False)
            ax_bwd_perm.spines['right'].set_visible(False)
            ax_bwd_perm.set_xlim(0, len(ppl_avg["Bwd"]) - 1)
            ax_bwd_perm.set_xticks([])
            ax_bwd_perm.set_ylim([10, 1e4])
            if size_idx == 0:
                ax_bwd_perm.set_ylabel("Bwd vs Perm\nln(perplexity)")

            # Plot Fwd - Bwd difference (row 3)
            ax_diff = axes[2, size_idx]
            if ppl_avg["Fwd"].size > 0 and ppl_avg["Bwd"].size > 0:
                diff = np.log(ppl_avg["Fwd"]) - np.log(ppl_avg["Bwd"])
                diff_std = np.sqrt(
                    ppl_std["Fwd"]**2 / ppl_avg["Fwd"]**2 + ppl_std["Bwd"]**2 / ppl_avg["Bwd"]**2
                )
                x = range(len(diff))
                ax_diff.plot(x, diff, color='#57D0DB', alpha=alpha, lw=lw, label="Fwd - Bwd")
                ax_diff.fill_between(
                    x, diff - diff_std, diff + diff_std, color='#57D0DB', alpha=0.2
                )
            ax_diff.spines['top'].set_visible(False)
            ax_diff.spines['right'].set_visible(False)
            ax_diff.plot([0, len(diff)], [0, 0], color='grey', lw=1, ls='--')
            ax_diff.set_xlim(0, len(diff) - 1)
            ax_diff.set_ylim([-0.1, 0.1])
            ax_diff.set_xlabel("Logging Steps")
            ax_diff.set_xticks([])
            if size_idx == 0:
                ax_diff.set_ylabel("Fwd - Bwd\nDifference")

        # Add legend to the top-right subplot
        axes[0, 2].legend()
        axes[1, 2].legend()
        axes[2, 2].legend()
        plt.subplots_adjust(left=0.15, right=0.99, top=0.92, bottom=0.08)
        plt.savefig(f"figs/{filename}")
        plt.close()

    # Create training figure
    create_figure("training", "train_losses_comparison.pdf")

    # Create validation figure
    create_figure("validation", "val_losses_comparison.pdf")

def main():
    # Get the wandb data
    get_wandb_data()

    # Plot the training and validation losses
    plot_train_val_losses()


if __name__ == "__main__":
    main()