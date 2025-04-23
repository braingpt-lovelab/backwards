import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import torch
import transformers
from sklearn.metrics.pairwise import cosine_similarity

model_pairs = {
    "GPT-2 (124M)": {
        "Fwd seed1 vs Fwd seed2": ["gpt2_scratch_neuro_tokenizer_bayes_fwd", "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed2"],
        "Fwd seed1 vs Fwd seed3": ["gpt2_scratch_neuro_tokenizer_bayes_fwd", "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed3"],
        "Fwd seed1 vs Rev seed1": ["gpt2_scratch_neuro_tokenizer_bayes_fwd", "gpt2_scratch_neuro_tokenizer_bayes_rev"],
        "Fwd seed1 vs Perm seed1": ["gpt2_scratch_neuro_tokenizer_bayes_fwd", "gpt2_scratch_neuro_tokenizer_bayes_perm"]
    },
    "GPT-2 (355M)": {
        "Fwd seed1 vs Fwd seed2": ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd", "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed2"],
        "Fwd seed1 vs Fwd seed3": ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd", "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed3"],
        "Fwd seed1 vs Rev seed1": ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd", "gpt2-medium_scratch_neuro_tokenizer_bayes_rev"],
        "Fwd seed1 vs Perm seed1": ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd", "gpt2-medium_scratch_neuro_tokenizer_bayes_perm"]
    },
    "GPT-2 (774M)": {
        "Fwd seed1 vs Fwd seed2": ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd", "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed2"],
        "Fwd seed1 vs Fwd seed3": ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd", "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed3"],
        "Fwd seed1 vs Rev seed1": ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd", "gpt2-large_scratch_neuro_tokenizer_bayes_rev"],
        "Fwd seed1 vs Perm seed1": ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd", "gpt2-large_scratch_neuro_tokenizer_bayes_perm"]
    }
}

plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})


def x_models_diffs(data_type="validation"):
    """
    Aggregate results plotter for all models.

    Plot the following results between model pairs:
        1. Model pair validation set PPL distributions
        2. Model pair validation set PPL T-test
        3. Model pair validation set PPL Pearson's r
        4. Model pair validation set PPL Cohen's d
    """
    n_rows = 3  # three sizes
    n_cols = 4  # (fwd,rev), (fwd,fwd2), (fwd,fwd3), (fwd,perm)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))

    for row_idx, (model_family, model_pair) in enumerate(model_pairs.items()):
        for col_idx, (model_pair_name, model_pair) in enumerate(model_pair.items()):
            model1_name, model2_name = model_pair
            print("-" * 20)
            print(f"{model_family}, {model_pair_name}")

            # Load results
            model1_ppls = np.load(
                f"model_results/{model1_name}/{type_of_abstract}/all_batches_ppl_{data_type}.npy"
            )
            model2_ppls = np.load(
                f"model_results/{model2_name}/{type_of_abstract}/all_batches_ppl_{data_type}.npy"
            )

            # Plot kde
            ax = axes[row_idx, col_idx]

            # Split `model_pair_name` by vs to get legend
            model1_legend, model2_legend = model_pair_name.split(" vs ")
            sns.kdeplot(
                model1_ppls,
                ax=ax,
                color='blue',
                alpha=0.5,
                label=model1_legend,
            )
            sns.kdeplot(
                model2_ppls,
                ax=ax,
                color='red',
                alpha=0.5,
                label=model2_legend,
            )
            ax.set_xlabel("Perplexity")
            ax.set_ylabel("Density")
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add model family name
            if col_idx == 0:
                ax.set_ylabel(model_family)
            else:
                ax.set_ylabel("")
            
            # Produce stats
            # Inc. T-test, Pearson's r, Cohen's d'
            t_stat, p_val = stats.ttest_rel(model1_ppls, model2_ppls)
            pearson_r = np.corrcoef(model1_ppls, model2_ppls)[0, 1]
            cohen_d = (np.mean(model1_ppls) - np.mean(model2_ppls)) / np.sqrt(
                np.var(model1_ppls - model2_ppls, ddof=1)
            )
            pearson_r = np.corrcoef(model1_ppls, model2_ppls)[0, 1]
            print(f"T-test: t_stat={t_stat}, p_val={p_val}")
            print(f"Pearson's r: {pearson_r}")
            print(f"Cohen's d: {cohen_d}")          # current, same as d'
            
    plt.tight_layout()
    plt.savefig("figs/model_pairs_ppl_diffs.png")


def main():
    x_models_diffs()


if __name__ == "__main__":
    results_dir = "model_results"
    figs_dir = "figs"
    type_of_abstract = "human_abstracts"

    main()
