import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from utils.model_utils import model_list

plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

"""
Compare GPT-2 variants train/val PPL magnitudes between forwards and backwards models.

1. PPL distributional difference of train ppl between forwards and backwards models.
2. PPL distributional difference of val ppl between forwards and backwards models.

Train/val ppl data format:
    e.g. 
    for GPT2, forwards_train_data_fpath = "data/gpt2_scratch_neuro_tokenizer_{train/val}.csv"
              backwards_train_data_fpath = "data/gpt2_scratch_neuro_tokenizer_backwards_{train/val}.csv"
    
    Columns of interest: `{Training/Validation} PPL`


Plotting:
    Figure layout: 
        Rows * 3: model of the same size (GPT2, GPT2-medium, GPT2-large)
        Columns * 2: distribution of train ppl and val ppl
    
    For each subplot, plot the distribution of train/val ppl for forwards and backwards models.
"""

def _load_ppl(
        forwards_model_fullname: str,
        backwards_model_fullname: str,
        data_type: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    forwards_data_fpath = f"{results_dir}/{forwards_model_fullname}_{data_type}.csv"
    backwards_data_fpath = f"{results_dir}/{backwards_model_fullname}_{data_type}.csv"

    with open(forwards_data_fpath, "r") as f:
        forwards_data = pd.read_csv(f)
    
    with open(backwards_data_fpath, "r") as f:
        backwards_data = pd.read_csv(f)

    # Get column ends with PPL and convert to numpy array
    forwards_ppl = forwards_data.filter(regex="PPL$").to_numpy().flatten()
    backwards_ppl = backwards_data.filter(regex="PPL$").to_numpy().flatten()
    # return forwards_ppl, backwards_ppl
    return np.log(forwards_ppl), np.log(backwards_ppl)


def ppl_distributional_diff():
    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = axes.flatten()

    model_pairs = []
    for k, v in model_list.items():
        if "gpt2" in k:
            model_pairs.append(model_list[k])

    for model_pair_idx, model_pair in enumerate(model_pairs):
        forwards_model_fullname = list(model_pair.keys())[0]
        backwards_model_fullname = list(model_pair.keys())[1]

        forwards_model_params = model_pair[forwards_model_fullname]
        backwards_model_params = model_pair[backwards_model_fullname]

        forwards_model_print_name = forwards_model_params["llm"]
        backwards_model_print_name = backwards_model_params["llm"]
        
        # Plot train ppl distribution
        forwards_train_ppl, backwards_train_ppl = \
            _load_ppl(forwards_model_fullname, backwards_model_fullname, "train")

        subplot_idx = model_pair_idx * n_cols
        sns.kdeplot(forwards_train_ppl, ax=axes[subplot_idx], label="Forwards")
        sns.kdeplot(backwards_train_ppl, ax=axes[subplot_idx], label="Backwards")
        axes[subplot_idx].hist(forwards_train_ppl, bins=20, alpha=0.2, color="blue", density=True)
        axes[subplot_idx].hist(backwards_train_ppl, bins=20, alpha=0.2, color="orange", density=True)
        axes[subplot_idx].set_title(f"{forwards_model_print_name}")
        axes[subplot_idx].spines['top'].set_visible(False)
        axes[subplot_idx].spines['right'].set_visible(False)
        axes[subplot_idx].set_xlim([1, 10])

        # T-test
        t_stat, p_val = stats.ttest_ind(forwards_train_ppl, backwards_train_ppl)
        print(f"{forwards_model_print_name} vs {backwards_model_print_name} train PPL")
        print(f"t({len(forwards_train_ppl)-1}) = {t_stat:.3f}, p = {p_val:.3f}")

        # Plot val ppl distribution
        forwards_val_ppl, backwards_val_ppl = \
            _load_ppl(forwards_model_fullname, backwards_model_fullname, "val")
        
        sns.kdeplot(forwards_val_ppl, ax=axes[model_pair_idx*n_cols+1], label="Forwards")
        sns.kdeplot(backwards_val_ppl, ax=axes[model_pair_idx*n_cols+1], label="Backwards")
        axes[model_pair_idx*n_cols+1].hist(forwards_val_ppl, bins=20, alpha=0.2, color="blue", density=True)
        axes[model_pair_idx*n_cols+1].hist(backwards_val_ppl, bins=20, alpha=0.2, color="orange", density=True)
        axes[model_pair_idx*n_cols+1].set_title(f"{forwards_model_print_name}")
        axes[model_pair_idx*n_cols+1].spines['top'].set_visible(False)
        axes[model_pair_idx*n_cols+1].spines['right'].set_visible(False)
        axes[model_pair_idx*n_cols+1].set_xlim([1, 10])

        # T-test
        t_stat, p_val = stats.ttest_ind(forwards_val_ppl, backwards_val_ppl)
        print(f"{forwards_model_print_name} vs {backwards_model_print_name} val PPL")
        print(f"t({len(forwards_val_ppl)-1}) = {t_stat:.3f}, p = {p_val:.3f}")

        axes[-2].set_xlabel("Training log(PPL)")
        axes[-1].set_xlabel("Validation log(PPL)")

    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/ppl_distributional_diff_train_val.pdf")


if __name__ == "__main__":
    results_dir = "data"
    figs_dir = "figs"
    ppl_distributional_diff()