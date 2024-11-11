import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from utils.model_utils import model_list

plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

"""
Compare BrainBench testcases PPL magnitudes between forwards and backwards models.

1. PPL distributional difference of correct options between forwards and backwards models.
2. PPL distributional difference of incorrect-correct options between forwards and backwards models.

Figure layout: 
    Rows * 3: model of the same size (GPT2, GPT2-medium, GPT2-large)
    Columns * 2: distribution of correct and incorrect-correct options
"""

def _ppl_correct_and_ppl_diff(PPL_A_and_B, labels):
    """
    Args:
        PPL_A_and_B: (N, 2) array of PPL values for A and B
        labels: (N, ) array of labels (0 or 1) where 0 indicates A correct, B incorrect

    Returns:
        PPL_correct: (N, ) array of PPL values of correct options
        PPL_diff: (N, ) array of PPL difference between incorrect and correct options
    """
    PPL_A = PPL_A_and_B[:, 0]
    PPL_B = PPL_A_and_B[:, 1]
    PPL_correct = []
    PPL_diff = []
    for i, label in enumerate(labels):
        if label == 0:  # A correct, B incorrect
            PPL_correct.append(PPL_A[i])
            PPL_diff.append(PPL_B[i] - PPL_A[i])
        else:
            PPL_correct.append(PPL_B[i])
            PPL_diff.append(PPL_A[i] - PPL_B[i])
    return np.array(PPL_correct), np.array(PPL_diff)


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
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
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

        # Left panel
        forwards_val_ppl, backwards_val_ppl = \
            _load_ppl(forwards_model_fullname, backwards_model_fullname, "validation")
    
        subplot_idx = model_pair_idx * n_cols
        sns.kdeplot(forwards_val_ppl, label="Forward", ax=axes[subplot_idx], color="blue")
        sns.kdeplot(backwards_val_ppl, label="Backward", ax=axes[subplot_idx], color="brown")
        axes[subplot_idx].hist(forwards_val_ppl, bins=20, alpha=0.1, color="blue", density=True)
        axes[subplot_idx].hist(backwards_val_ppl, bins=20, alpha=0.1, color="brown", density=True)
        axes[subplot_idx].set_title(f"{forwards_model_print_name}")
        axes[subplot_idx].spines['top'].set_visible(False)
        axes[subplot_idx].spines['right'].set_visible(False)
        axes[subplot_idx].set_xlim([0, 170])
        axes[subplot_idx].set_ylim([0, 0.08])
        axes[subplot_idx].grid(True)

        # Right panel
        forwards_dir = f"{results_dir}/{forwards_model_fullname}/{type_of_abstract}"
        backwards_dir = f"{results_dir}/{backwards_model_fullname}/{type_of_abstract}"
        forwards_PPL_A_and_B = np.load(f"{forwards_dir}/PPL_A_and_B.npy")
        forwards_labels = np.load(f"{forwards_dir}/labels.npy")
        backwards_PPL_A_and_B = np.load(f"{backwards_dir}/PPL_A_and_B.npy")
        backwards_labels = np.load(f"{backwards_dir}/labels.npy")

        forwards_PPL_correct, forwards_PPL_diff = \
            _ppl_correct_and_ppl_diff(forwards_PPL_A_and_B, forwards_labels)
        backwards_PPL_correct, backwards_PPL_diff = \
            _ppl_correct_and_ppl_diff(backwards_PPL_A_and_B, backwards_labels)

        subplot_idx = model_pair_idx * n_cols + 1
        sns.kdeplot(forwards_PPL_correct, label="Forward", ax=axes[subplot_idx], color="blue")
        sns.kdeplot(backwards_PPL_correct, label="Backward", ax=axes[subplot_idx], color="brown")
        axes[subplot_idx].hist(forwards_PPL_correct, bins=20, alpha=0.1, color="blue", density=True)
        axes[subplot_idx].hist(backwards_PPL_correct, bins=20, alpha=0.1, color="brown", density=True)
        axes[subplot_idx].set_title(f"{forwards_model_print_name}")
        axes[subplot_idx].spines['top'].set_visible(False)
        axes[subplot_idx].spines['right'].set_visible(False)
        axes[subplot_idx].set_xlim([0, 170])
        axes[subplot_idx].set_ylim([0, 0.08])
        axes[subplot_idx].set_ylabel("")
        axes[subplot_idx].grid(True)
        
        # T-test
        t_stat, p_val = stats.ttest_rel(forwards_PPL_correct, backwards_PPL_correct)
        print(f"Correct PPL: {forwards_model_print_name} vs {backwards_model_print_name}")
        print(f"t({len(forwards_PPL_correct) - 1}) = {t_stat:.3f}, p = {p_val:.3f}")
              
        axes[-2].set_xlabel("PPL (Validation)")
        axes[-1].set_xlabel("PPL (BrainBench Correct)")

    axes[0].annotate("A", xy=(-0.2, 1.2), xycoords="axes fraction", fontsize=20, fontweight="bold")
    axes[1].annotate("B", xy=(-0.2, 1.2), xycoords="axes fraction", fontsize=20, fontweight="bold")
    axes[-1].legend() 
    plt.tight_layout()
    plt.savefig(f"{figs_dir}/ppl_distributional_diff_val_and_test.pdf")


if __name__ == "__main__":
    results_dir = "model_results"
    figs_dir = "figs"
    type_of_abstract = "human_abstracts"

    ppl_distributional_diff()
