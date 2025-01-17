import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from utils.model_utils import model_list
from utils.general_utils import str2bool
from utils.general_utils import scorer_acc, scorer_sem, scorer_ppl_diff


def get_llm_accuracies(model_results_dir, use_human_abstract=True):
    llms = model_list
    # Remove all keys except "gpt2"
    # To plot only the GPT-2 family models.
    llms = {k: v for k, v in llms.items() if "gpt2" in k}

    all_data = []  # export as csv for R analysis

    for i, llm_family in enumerate(llms.keys()):
        for j, llm in enumerate(llms[llm_family]):
            print(f"Processing {llm_family}, {llm}")
            if use_human_abstract:
                type_of_abstract = 'human_abstracts'
            else:
                type_of_abstract = 'llm_abstracts'

            results_dir = os.path.join(
                f"{model_results_dir}/{llm.replace('/', '--')}/{type_of_abstract}"
            )

            PPL_fname = "PPL_A_and_B"
            label_fname = "labels"
            PPL_A_and_B = np.load(f"{results_dir}/{PPL_fname}.npy")
            labels = np.load(f"{results_dir}/{label_fname}.npy")

            acc = scorer_acc(PPL_A_and_B, labels)
            sem = scorer_sem(PPL_A_and_B, labels)
            ppl_diff = scorer_ppl_diff(PPL_A_and_B, labels)
            llms[llm_family][llm]["acc"] = acc
            llms[llm_family][llm]["sem"] = sem
            llms[llm_family][llm]["ppl_diff"] = ppl_diff

            # Process each item
            for item_index in range(labels.shape[0]):
                # Create a data point for this item
                data_point = {
                    'model_id': i * len(llms[llm_family]) + j + 1,
                    'direction': 0 if "backward" in llm else 1,
                    'model_size': 2 if "medium" in llm else (3 if "large" in llm else 1),
                    'item': item_index,
                    'correct': int(np.argmin(PPL_A_and_B[item_index]) == labels[item_index])
                }
                all_data.append(data_point)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save the data for R analysis
    df.to_csv("model_performance_x_direction_x_size_x_item.csv", index=False)
    print("Data saved as model_performance_x_direction_x_size_x_item.csv")

    return llms


def get_human_accuracies(use_human_abstract):
    """
    Overall accuracy (based on `correct` column) for `human` created cases
    """
    # Read data
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")
    if use_human_abstract:
        who = "human"
    else:
        who = "machine"

    correct = 0
    total = 0
    for _, row in df.iterrows():
        if row["journal_section"].startswith(who):
            correct += row["correct"]
            total += 1
    acc = correct / total
    sem = np.sqrt(acc * (1 - acc) / total)
    return acc, sem


def get_human_accuracies_top_expertise(use_human_abstract, top_pct=0.2):
    """
    Overall accuracy (based on `correct` column) for `human` created cases,
    but for each abstract_id, only uses experts with top 20% rated expertise
    """
    # Read data
    df = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")
    
    if use_human_abstract:
        who = "human"
    else:
        who = "machine"

    # Group by abstract_id and journal_section that starts with `who`
    # Then, for each abstract_id, only use experts with top 20% rated expertise
    df_grouped = df[df["journal_section"].str.startswith(who)].groupby("abstract_id")
    df_grouped = df_grouped.apply(
        lambda x: x.nlargest(int(len(x)*top_pct), "expertise")
    )
    df_grouped = df_grouped.reset_index(drop=True)

    correct = 0
    total = 0
    for _, row in df_grouped.iterrows():
        correct += row["correct"]
        total += 1
    acc = correct / total
    sem = np.sqrt(acc * (1 - acc) / total)
    return acc, sem


def plot(use_human_abstract):
    """
    Plot LLMs vs human experts.

    1) Plot accuracy of each llm as a bar. 
    Bar height is accuracy, bar groups by llm family.
    Bar color and hatch follow keys in `llms` dict.

    2) Plot human experts as a horizontal line
    """
    llms = get_llm_accuracies(model_results_dir, use_human_abstract)

    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
    fig, ax = plt.subplots(figsize=(8, 6))

    # llms
    all_llm_accuracies = []
    all_llm_sems = []
    all_llm_ppl_diffs = []
    all_llm_names = []
    all_llm_colors = []
    all_llm_hatches = []
    all_llm_xticks = []
    all_llm_alphas = []

    for family_index, llm_family in enumerate(llms.keys()):
        for llm in llms[llm_family]:
            all_llm_accuracies.append(llms[llm_family][llm]["acc"])
            all_llm_sems.append(llms[llm_family][llm]["sem"])
            all_llm_ppl_diffs.append(llms[llm_family][llm]["ppl_diff"])
            all_llm_names.append(llms[llm_family][llm]["llm"])
            all_llm_colors.append(llms[llm_family][llm]["color"])
            all_llm_hatches.append(llms[llm_family][llm]["hatch"])
            all_llm_alphas.append(llms[llm_family][llm]["alpha"])
            # # Anchor on `family_index`
            # # llm within a family should be spaced out smaller than between families
            all_llm_xticks.append(family_index + len(all_llm_xticks))
    
    print(all_llm_xticks)

    # Bar
    for i in range(len(all_llm_xticks)):
        ax.bar(
            all_llm_xticks[i],
            all_llm_accuracies[i],
            yerr=all_llm_sems[i],
            color=all_llm_colors[i],
            hatch=all_llm_hatches[i],
            alpha=all_llm_alphas[i],
            label=all_llm_names[i],
            edgecolor='k',
            capsize=3
        )
    
    ax.legend(all_llm_names, loc='upper left')

    # human
    # plot as horizontal line
    human_acc, human_sem = get_human_accuracies(use_human_abstract)
    hline = ax.axhline(y=human_acc, color='b', linestyle='--', lw=3)
    ax.fill_between(
        [hline.get_xdata()[0], all_llm_xticks[-1]+1],
        human_acc - human_sem,
        human_acc + human_sem,
        color='b',
        alpha=0.3
    )

    print('human_acc:', human_acc)
    human_acc_top_expertise, _ = get_human_accuracies_top_expertise(use_human_abstract)
    print('human_acc_top_expertise:', human_acc_top_expertise)

    # Add annotations (Human expert)
    # In the middle of the plot, below the horizontal line
    ax.text(
        all_llm_xticks[-1]+0.5,
        human_acc+0.03,
        "Human\nexperts",
        fontsize=16,
        color='k'
    )

    ax.set_ylabel("Accuracy")
    ax.set_ylim([0.5, 1])
    # ax.set_xlim([None, all_llm_xticks[-1]+1])
    ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if use_human_abstract:
        plt.savefig(f"{base_fname}_human_abstract.pdf")
        plt.savefig(f"{base_fname}_human_abstract.svg")
    else:
        plt.savefig(f"{base_fname}_llm_abstract.pdf")

    # Significance testing 
    # 1. Compare forwards vs backwards models acc
    fwd_models = []
    bwd_models = []
    for family_index, llm_family in enumerate(llms.keys()):
        llm_names = list(llms[llm_family].keys())
        
        llm_fwd_acc = llms[llm_family][llm_names[0]]["acc"]
        llm_bwd_acc = llms[llm_family][llm_names[1]]["acc"]
        fwd_models.append(llm_fwd_acc)
        bwd_models.append(llm_bwd_acc)

    print(fwd_models, bwd_models)
    t_stat, p_val = stats.ttest_rel(fwd_models, bwd_models)
    print(f"t({len(fwd_models)-1}) = {t_stat:.3f}, p = {p_val:.3f}")

    # 2. Repeated AVONA for model size and training direction
    # Done in `anova_stats.R`

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str2bool, default=True)

    model_results_dir = "model_results"
    human_results_dir = "human_results"
    base_fname = "figs/overall_accuracy_model_vs_human"
    plot(parser.parse_args().use_human_abstract)
