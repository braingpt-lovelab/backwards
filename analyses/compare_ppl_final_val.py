import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from utils.model_utils import model_list

plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

"""
Compare GPT-2 variants val PPL magnitudes between forwards and backwards models.

val ppl data format:
    e.g. 
    for GPT2, forwards_train_data_fpath = "model_results/{model}/{human_abstracts}/all_batches_ppl_val.npy"
    where the numpy array contains ppl of all validation items.
    
Plotting:
    Figure layout: 
        Rows * 3: model of the same size (GPT2, GPT2-medium, GPT2-large)
        Columns * 1: distribution of val ppl
"""

def _load_ppl(
        forwards_model_fullname: str,
        backwards_model_fullname: str,
        data_type: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    forwards_data_fpath = f"model_results/{forwards_model_fullname}/{type_of_abstract}/all_batches_ppl_{data_type}.npy"
    backwards_data_fpath = f"model_results/{backwards_model_fullname}/{type_of_abstract}/all_batches_ppl_{data_type}.npy"

    forwards_data = np.load(forwards_data_fpath)
    backwards_data = np.load(backwards_data_fpath)
    return forwards_data, backwards_data


def ppl_distributional_diff():
    n_rows = 3
    n_cols = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 8))
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
        
        # Plot final val ppl distribution
        forwards_val_ppl, backwards_val_ppl = \
            _load_ppl(forwards_model_fullname, backwards_model_fullname, "validation")

        subplot_idx = model_pair_idx * n_cols
        sns.kdeplot(forwards_val_ppl, ax=axes[subplot_idx], label="Forwards")
        sns.kdeplot(backwards_val_ppl, ax=axes[subplot_idx], label="Backwards")
        axes[subplot_idx].hist(forwards_val_ppl, bins=20, alpha=0.2, color="blue", density=True)
        axes[subplot_idx].hist(backwards_val_ppl, bins=20, alpha=0.2, color="orange", density=True)
        axes[subplot_idx].set_title(f"{forwards_model_print_name}")
        axes[subplot_idx].spines['top'].set_visible(False)
        axes[subplot_idx].spines['right'].set_visible(False)
        axes[subplot_idx].set_xlim([0, 200])
        axes[subplot_idx].set_ylim([0, 0.1])

        # T-test
        t_stat, p_val = stats.ttest_ind(forwards_val_ppl, backwards_val_ppl)
        print(f"{forwards_model_print_name} vs {backwards_model_print_name} val PPL")
        dof = len(forwards_val_ppl) + len(backwards_val_ppl) - 2
        print(f"t({dof}) = {t_stat:.3f}, p = {p_val:.3f}")

        axes[-1].set_xlabel("Validation PPL")

    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/ppl_distributional_diff_final_val.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str, default="True")

    if parser.parse_args().use_human_abstract == "True":
        use_human_abstract = True
    else:
        use_human_abstract = False

    if use_human_abstract:
        type_of_abstract = 'human_abstracts'
    else:
        type_of_abstract = 'llm_abstracts'
    
    figs_dir = "figs"
    ppl_distributional_diff()