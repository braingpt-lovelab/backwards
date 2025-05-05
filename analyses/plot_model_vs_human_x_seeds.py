import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from utils.model_utils import model_list
from utils.general_utils import str2bool
from utils.general_utils import scorer_acc, scorer_sem, scorer_ppl_diff


comparison = {
    "GPT-2 (124M)": {
        "Fwd": [
            "gpt2_scratch_neuro_tokenizer_bayes_fwd", 
            "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed2",
            "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed3",
        ],
        "Bwd": [
            "gpt2_scratch_neuro_tokenizer_bayes_rev", 
            "gpt2_scratch_neuro_tokenizer_bayes_rev_seed2",
            "gpt2_scratch_neuro_tokenizer_bayes_rev_seed3",
        ]
    },
    "GPT-2 (355M)": {
        "Fwd": [
            "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed2",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed3",
        ],
        "Bwd": [
            "gpt2-medium_scratch_neuro_tokenizer_bayes_rev",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed2",
            "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed3",
        ]
    },
    "GPT-2 (774M)": {
        "Fwd": [
            "gpt2-large_scratch_neuro_tokenizer_bayes_fwd",
            "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed2",
            "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed3",
        ],
        "Bwd": [
            "gpt2-large_scratch_neuro_tokenizer_bayes_rev",
            "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed2",
            "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed3",
        ]
    },
}

comparison_styles = {
    "GPT-2 (124M)": {
        "Fwd": {
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "",
        },
        "Bwd": {
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "\\",
        }
    },
    "GPT-2 (355M)": {
        "Fwd": {
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "",
        },
        "Bwd": {
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "\\",
        }
    },
    "GPT-2 (774M)": {
        "Fwd": {
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "",
        },
        "Bwd": {
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "\\",
        } 
    }
}

def get_llm_accuracies(model_results_dir, use_human_abstract=True):
    accuracies = {}
    for model_family, directions in comparison.items():
        accuracies[model_family] = {}
        for direction, models in directions.items():
            # One direction points to models x seeds
            accuracies[model_family][direction] = {}
            per_direction_models_accs = []
            for model in models:
                print(f"Processing {model_family}, {direction}, {model}")
                if use_human_abstract:
                    type_of_abstract = 'human_abstracts'
                else:
                    type_of_abstract = 'llm_abstracts'

                results_dir = os.path.join(
                    f"{model_results_dir}/{model.replace('/', '--')}/{type_of_abstract}"
                )

                PPL_fname = "PPL_A_and_B"
                label_fname = "labels"
                PPL_A_and_B = np.load(f"{results_dir}/{PPL_fname}.npy")
                labels = np.load(f"{results_dir}/{label_fname}.npy")

                acc = scorer_acc(PPL_A_and_B, labels)
                print(f"Accuracy: {acc}")
                per_direction_models_accs.append(acc)
            accuracies[model_family][direction]["acc"] = np.mean(per_direction_models_accs)
            accuracies[model_family][direction]["sem"] = stats.sem(per_direction_models_accs)
    return accuracies


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


def plot(use_human_abstract):
    llm_accuracies = get_llm_accuracies(model_results_dir, use_human_abstract)

    plt.rcParams.update({'font.size': 12, 'font.weight': 'normal'})
    fig, ax = plt.subplots(figsize=(6, 4))

    # Bar plot parameters
    model_families = list(comparison.keys())
    directions = ['Fwd', 'Bwd']
    bar_width = 0.35  # Width of each bar
    group_width = 1.0  # Width of each model family group
    spacing = 0.1      # Spacing between groups

    # Calculate x positions for bars
    x_positions = []
    for i in range(len(model_families)):
        start = i * (group_width + spacing)
        x_positions.append([start, start + bar_width])
    x_positions = np.array(x_positions)

    # Plot bars for each model family and direction
    for i, model_family in enumerate(model_families):
        for j, direction in enumerate(directions):
            acc = llm_accuracies[model_family][direction]['acc']
            sem = llm_accuracies[model_family][direction]['sem']
            style = comparison_styles[model_family][direction]
            label = f"{model_family} ({direction})"
            
            # Plot bar
            ax.bar(
                x_positions[i, j],
                acc,
                bar_width,
                yerr=sem,
                color=style['color'],
                alpha=style['alpha'],
                hatch=style['hatch'],
                edgecolor='black',
                capsize=5,
                label=label
            )
            ax.legend()

    # Human accuracy line
    human_acc, human_sem = get_human_accuracies(use_human_abstract)
    hline = ax.axhline(y=human_acc, color='b', linestyle='--', lw=3)
    ax.fill_between(
        [x_positions[0, 0] - 0.5, x_positions[-1, -1] + 0.5],
        human_acc - human_sem,
        human_acc + human_sem,
        color='b',
        alpha=0.3
    )
    print('human_acc:', human_acc)

    # Add human expert annotation
    ax.text(
        x_positions[-1, -1] + 0.5,
        human_acc + 0.03,
        "Human\nexperts",
        fontsize=12,
        color='k',
        horizontalalignment='left',
        verticalalignment='bottom'
    )

    # Customize plot
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0.5, 1])
    ax.set_xlim([x_positions[0, 0] - 0.5, x_positions[-1, -1] + 0.5])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if use_human_abstract:
        plt.savefig(f"{base_fname}_human_abstract.pdf")
        plt.savefig(f"{base_fname}_human_abstract.svg")
    else:
        plt.savefig(f"{base_fname}_llm_abstract.pdf")
        plt.savefig(f"{base_fname}_llm_abstract.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str2bool, default=True)

    model_results_dir = "model_results"
    human_results_dir = "human_results"
    base_fname = "figs/overall_accuracy_model_vs_human"
    plot(parser.parse_args().use_human_abstract)
