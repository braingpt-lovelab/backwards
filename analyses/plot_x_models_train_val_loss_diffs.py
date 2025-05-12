import os
import pickle
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

    # Define model sizes and other parameters
    model_sizes = ["GPT-2 (124M)", "GPT-2 (355M)", "GPT-2 (774M)"]
    seeds = ["seed1", "seed2", "seed3"]
    model_labels = ["Fwd", "Bwd", "Perm"]
    colors = ['#E8B7D4', '#FF7B89', '#5874DC']  # Colors for Fwd-Bwd, Fwd-Perm, Bwd-Perm
    alpha = 1
    lw = plt.rcParams['lines.linewidth'] ** 2
    line_styles = ['-', '--', ':']

    # Function to create a figure for either train or validation
    def create_figure(data_type, filename):
        fig = plt.figure(figsize=(8, 3))
        gs = gridspec.GridSpec(1, 3, wspace=0.1)
        axes = [plt.subplot(gs[0, col]) for col in range(3)]

        # Store handles and labels for the legend
        legend_handles = []
        legend_labels = []

        for size_idx, model_size in enumerate(model_sizes):
            ax = axes[size_idx]
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

            # Plot differences and collect handles/labels only once (from the first subplot)
            if size_idx == 0:
                if ppl_avg["Fwd"].size > 0 and ppl_avg["Bwd"].size > 0:
                    diff_fwd_bwd = ppl_avg["Fwd"] - ppl_avg["Bwd"]
                    diff_fwd_bwd_std = np.sqrt(ppl_std["Fwd"]**2 + ppl_std["Bwd"]**2)
                    x = range(len(diff_fwd_bwd))
                    line, = ax.plot(x, diff_fwd_bwd, color=colors[0], alpha=alpha, lw=lw, label="Fwd - Bwd", linestyle=line_styles[0])
                    ax.fill_between(
                        x, diff_fwd_bwd - diff_fwd_bwd_std, diff_fwd_bwd + diff_fwd_bwd_std,
                        color=colors[0], alpha=0.2
                    )
                    legend_handles.append(line)
                    legend_labels.append("Fwd - Bwd")

                if ppl_avg["Fwd"].size > 0 and ppl_avg["Perm"].size > 0:
                    diff_fwd_perm = ppl_avg["Fwd"] - ppl_avg["Perm"]
                    diff_fwd_perm_std = np.sqrt(ppl_std["Fwd"]**2 + ppl_std["Perm"]**2)
                    x = range(len(diff_fwd_perm))
                    line, = ax.plot(x, diff_fwd_perm, color=colors[1], alpha=alpha, lw=lw, label="Fwd - Perm", linestyle=line_styles[1])
                    ax.fill_between(
                        x, diff_fwd_perm - diff_fwd_perm_std, diff_fwd_perm + diff_fwd_perm_std,
                        color=colors[1], alpha=0.2
                    )
                    legend_handles.append(line)
                    legend_labels.append("Fwd - Perm")

                if ppl_avg["Bwd"].size > 0 and ppl_avg["Perm"].size > 0:
                    diff_bwd_perm = ppl_avg["Bwd"] - ppl_avg["Perm"]
                    diff_bwd_perm_std = np.sqrt(ppl_std["Bwd"]**2 + ppl_std["Perm"]**2)
                    x = range(len(diff_bwd_perm))
                    line, = ax.plot(x, diff_bwd_perm, color=colors[2], alpha=alpha, lw=lw, label="Bwd - Perm", linestyle=line_styles[2])
                    ax.fill_between(
                        x, diff_bwd_perm - diff_bwd_perm_std, diff_bwd_perm + diff_bwd_perm_std,
                        color=colors[2], alpha=0.2
                    )
                    legend_handles.append(line)
                    legend_labels.append("Bwd - Perm")
            else:
                # Plot without labels for other subplots to avoid duplicate legend entries
                if ppl_avg["Fwd"].size > 0 and ppl_avg["Bwd"].size > 0:
                    diff_fwd_bwd = ppl_avg["Fwd"] - ppl_avg["Bwd"]
                    diff_fwd_bwd_std = np.sqrt(ppl_std["Fwd"]**2 + ppl_std["Bwd"]**2)
                    x = range(len(diff_fwd_bwd))
                    ax.plot(x, diff_fwd_bwd, color=colors[0], alpha=alpha, lw=lw, linestyle=line_styles[0])
                    ax.fill_between(
                        x, diff_fwd_bwd - diff_fwd_bwd_std, diff_fwd_bwd + diff_fwd_bwd_std,
                        color=colors[0], alpha=0.2
                    )

                if ppl_avg["Fwd"].size > 0 and ppl_avg["Perm"].size > 0:
                    diff_fwd_perm = ppl_avg["Fwd"] - ppl_avg["Perm"]
                    diff_fwd_perm_std = np.sqrt(ppl_std["Fwd"]**2 + ppl_std["Perm"]**2)
                    x = range(len(diff_fwd_perm))
                    ax.plot(x, diff_fwd_perm, color=colors[1], alpha=alpha, lw=lw, linestyle=line_styles[1])
                    ax.fill_between(
                        x, diff_fwd_perm - diff_fwd_perm_std, diff_fwd_perm + diff_fwd_perm_std,
                        color=colors[1], alpha=0.2
                    )

                if ppl_avg["Bwd"].size > 0 and ppl_avg["Perm"].size > 0:
                    diff_bwd_perm = ppl_avg["Bwd"] - ppl_avg["Perm"]
                    diff_bwd_perm_std = np.sqrt(ppl_std["Bwd"]**2 + ppl_std["Perm"]**2)
                    x = range(len(diff_bwd_perm))
                    ax.plot(x, diff_bwd_perm, color=colors[2], alpha=alpha, lw=lw, linestyle=line_styles[2])
                    ax.fill_between(
                        x, diff_bwd_perm - diff_bwd_perm_std, diff_bwd_perm + diff_bwd_perm_std,
                        color=colors[2], alpha=0.2
                    )
                    
            ax.set_title(f"{model_size}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(0, len(ppl_avg["Fwd"]) - 1)
            ax.set_yscale('symlog')  # Set y-axis to symmetric log scale
            ax.set_ylim([-1e4, 10])
            ax.set_xlabel("Logging Steps")
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.plot([0, len(ppl_avg["Fwd"]) - 1], [0, 0], color='grey', lw=2, linestyle='--')
            if size_idx == 0:
                ax.set_ylabel("Perplexity Difference")
            if size_idx != 0:
                ax.set_yticklabels([])

        # Adjust layout to make space for the legend at the bottom
        plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.25)

        # Add a single legend at the bottom with 3 columns
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            ncol=3,
            bbox_to_anchor=(0.5, 0.008),
            frameon=True
        )

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