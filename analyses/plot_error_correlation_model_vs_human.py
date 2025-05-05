import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils import model_utils


def average_acc_across_cases_participants(who="human"):
    """
    Given all test cases, go through each case's average accuracy
    across all participants.

    Iterate through `online_study_data`, build a dict to accumulate for 
    each abstract_id's n_correct and total occurrences.

    And given `who`, only consider `journal_section` startswith 
    `who`.

    Intermediate output:
        {"abstract_id": [n_correct, total], ...}
    
    return:
        {"doi": avg_acc, ...}
    """
    acc_dict = {}
    for _, row in online_study_data.iterrows():
        abstract_id = int(float(row["abstract_id"]))
        journal_section = row["journal_section"]
        if not journal_section.startswith(who):
            continue

        if abstract_id not in acc_dict:
            acc_dict[abstract_id] = [row["correct"], 1]
        else:
            acc_dict[abstract_id][0] += row["correct"]
            acc_dict[abstract_id][1] += 1
        
    # Sanity check: show largest total and smallest total
    largest_total = 0
    smallest_total = 100000
    for abstract_id, acc_list in acc_dict.items():
        total = acc_list[1]
        if total > largest_total:
            largest_total = total
        if total < smallest_total:
            smallest_total = total
    print(f"largest_total: {largest_total}")
    print(f"smallest_total: {smallest_total}")

    # Compute average accuracy for each abstract_id
    # but using doi as key
    avg_acc_dict = {}
    for abstract_id, acc_list in acc_dict.items():
        doi = mapper_dict[abstract_id]
        avg_acc = acc_list[0] / acc_list[1]
        avg_acc_dict[doi] = avg_acc
    return avg_acc_dict


def ppl_diffs_for_single_model(llm, who="human"):
    """
    For each test case, we compute ppl_diff in terms of PPL_incorrect - PPL_correct,
    which need to use `labels` to determine which is incorrect and which is correct.

    Then, we convert individual ppl_diff into ranks and pass on for spearmanr.
    """
    ppl_diffs_dict = {}
    if who == "human":
        abstract_type = "human_abstracts"
    else:
        abstract_type = "llm_abstracts"

    PPL_A_and_B = np.load(f"{model_results_dir}/{llm.replace('/', '--')}/{abstract_type}/PPL_A_and_B.npy")
    labels = np.load(f"{model_results_dir}/{llm.replace('/', '--')}/{abstract_type}/labels.npy")
    for i, label in enumerate(labels):
        doi = human_abstracts_dois[i] if who == "human" else machine_abstracts_dois[i]
        
        if label == 0: # incorrect is B
            ppl_diff = PPL_A_and_B[i][1] - PPL_A_and_B[i][0]
        else:       # incorrect is A
            ppl_diff = PPL_A_and_B[i][0] - PPL_A_and_B[i][1]

        ppl_diffs_dict[doi] = ppl_diff
     
    return ppl_diffs_dict


def per_model_human_error_corr(avg_acc_dict_machine, avg_acc_dict_human):
    """
    Given avg_acc_dict_machine and avg_acc_dict_human,
    compute the correlation between the two by iterating 
    the dois and compiling the two lists of avg_acc. Finally
    compute the correlation between the two lists.
    """
    dois = []
    avg_acc_machine = []
    avg_acc_human = []
    for doi in avg_acc_dict_machine.keys():
        dois.append(doi)
        avg_acc_machine.append(avg_acc_dict_machine[doi])
        avg_acc_human.append(avg_acc_dict_human[doi])
    
    # Convert both lists to ranks
    avg_acc_machine = stats.rankdata(avg_acc_machine)
    avg_acc_human = stats.rankdata(avg_acc_human)

    # Compute correlation
    corr, p = stats.spearmanr(avg_acc_machine, avg_acc_human)
    print(f"corr: {corr}, p: {p}")
    return corr, p, dois, avg_acc_machine, avg_acc_human


def per_model_model_error_corr(avg_acc_dict_machine_1, avg_acc_dict_machine_2):
    """
    Given avg_acc_dict_machine_1 and avg_acc_dict_machine_2,
    compute the correlation between the two by iterating
    the dois and compiling the two lists of avg_acc. Finally
    compute the correlation between the two lists.
    """
    dois = []
    avg_acc_machine_1 = []
    avg_acc_machine_2 = []
    for doi in avg_acc_dict_machine_1.keys():
        dois.append(doi)
        avg_acc_machine_1.append(avg_acc_dict_machine_1[doi])
        avg_acc_machine_2.append(avg_acc_dict_machine_2[doi])
    
    # Convert both lists to ranks
    avg_acc_machine_1 = stats.rankdata(avg_acc_machine_1)
    avg_acc_machine_2 = stats.rankdata(avg_acc_machine_2)

    # Compute correlation
    corr, p = stats.spearmanr(avg_acc_machine_1, avg_acc_machine_2)
    print(f"corr: {corr}, p: {p}")
    return corr, p, dois, avg_acc_machine_1, avg_acc_machine_2


def get_all_correlations(who, avg_acc_dict_human):
    # Flatten `llms` into a list of llms
    llms_flat = []
    llms_flat_names = []
    for llm_family in llms.keys():
        for llm in llms[llm_family].keys():
            llms_flat.append(llm)
            llms_flat_names.append(llms[llm_family][llm]["llm"])
        
    corr_all_models_and_human = np.zeros((len(llms_flat)+1, len(llms_flat)+1))
    for i in range(len(corr_all_models_and_human)):
        for j in range(len(corr_all_models_and_human)):
            # Skip upper triangle (inc. diagonal)
            if i <= j:
                continue

            # last row (human vs machines)
            if i == len(corr_all_models_and_human) - 1:
                avg_acc_dict_human = avg_acc_dict_human
                avg_acc_dict_machine = ppl_diffs_for_single_model(llms_flat[j], who=who)
                corr, _, _, _, _ = per_model_human_error_corr(
                    avg_acc_dict_machine=avg_acc_dict_machine,
                    avg_acc_dict_human=avg_acc_dict_human,
                )

            # all other entries (machine vs machine)
            else:
                avg_acc_dict_machine_1 = ppl_diffs_for_single_model(llms_flat[i], who=who)
                avg_acc_dict_machine_2 = ppl_diffs_for_single_model(llms_flat[j], who=who)
                corr, _, _, _, _ = per_model_model_error_corr(
                    avg_acc_dict_machine_1=avg_acc_dict_machine_1,
                    avg_acc_dict_machine_2=avg_acc_dict_machine_2,
                )

            corr_all_models_and_human[i][j] = corr
    
    # Create a mask for the upper triangle
    # and set the diagonal to NaN
    mask = np.triu(np.ones_like(corr_all_models_and_human, dtype=bool), k=0)
    corr_all_models_and_human[mask] = np.nan
    return corr_all_models_and_human, llms_flat_names


# def plot_model_model_and_model_human_correlation(corr_all_models_and_human, llms_flat_names, who):
#     plt.rcParams.update({"font.size": 12, "font.weight": "normal"})
#     avg_model_model_corr = np.nanmean(corr_all_models_and_human[:-1, :-1])
#     std_model_model_corr = np.nanstd(corr_all_models_and_human[:-1, :-1])
#     avg_model_human_corr = np.nanmean(corr_all_models_and_human[-1, :-1])
#     std_model_human_corr = np.nanstd(corr_all_models_and_human[-1, :-1])

#     print(f"Average model-model correlation: {avg_model_model_corr:.2f}")
#     print(f"Std model-model correlation: {std_model_model_corr:.2f}")
#     print(f"Average model-human correlation: {avg_model_human_corr:.2f}")
#     print(f"Std model-human correlation: {std_model_human_corr:.2f}")
#     print("-" * 50)

#     # Plot barplots for model-model and model-human correlations
#     fig, ax = plt.subplots(figsize=(3, 2))
#     bar_width = 0.35
#     ax.bar(
#         np.arange(2),
#         [avg_model_model_corr, avg_model_human_corr],
#         bar_width,
#         yerr=[std_model_model_corr, std_model_human_corr],
#         capsize=5,
#         color=["blue", "orange"],
#         tick_label=['Model-Model', 'Model-Human']
#     )
#     ax.set_ylabel('Correlation')
#     ax.set_ylim(0, 1)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.grid(axis='y', linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(f"figs/error_correlation_mm_vs_mh_{who}_created.pdf")
#     plt.close()


# def plot_human_vs_model_correlation(corr_all_models_and_human, who):
#     plt.rcParams.update({"font.size": 12, "font.weight": "normal"})

#     # Compute average and std correlation for each model size and direction
#     # corr_all_models_and_human[-1, :-1] is human vs models, and 
#     # is organized as:
#     # [small_fwd_seed1, small_fwd_seed2, small_fwd_seed3,
#     #  small_bwd_seed1, small_bwd_seed2, small_bwd_seed3,
#     #  medium_fwd_seed1, medium_fwd_seed2, medium_fwd_seed3,
#     #  medium_bwd_seed1, medium_bwd_seed2, medium_bwd_seed3,
#     #  large_fwd_seed1, large_fwd_seed2, large_fwd_seed3,
#     #  large_bwd_seed1, large_bwd_seed2, large_bwd_seed3]
#     avg_corrs = {}
#     std_corrs = {}
#     sizes = ["small", "medium", "large"]
#     indices = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18)]

#     # Collect for stats testing
#     all_fwd_vs_human = []
#     all_bwd_vs_human = []
#     for size, (fwd_idx, bwd_idx) in zip(sizes, zip(indices[::2], indices[1::2])):
#         print(f"\n{size} models")
#         print("-" * 50)
#         print(f"Forward models: {fwd_idx[0]}-{fwd_idx[1]}")
#         print(f"Backward models: {bwd_idx[0]}-{bwd_idx[1]}")
#         all_fwd_vs_human_entries = corr_all_models_and_human[-1, :-1][fwd_idx[0]:fwd_idx[1]]
#         all_bwd_vs_human_entries = corr_all_models_and_human[-1, :-1][bwd_idx[0]:bwd_idx[1]]
#         all_fwd_vs_human.extend(all_fwd_vs_human_entries)
#         all_bwd_vs_human.extend(all_bwd_vs_human_entries)
#         avg_corrs[f"{size}_fwd"] = np.mean(all_fwd_vs_human_entries)
#         std_corrs[f"{size}_fwd"] = np.std(all_fwd_vs_human_entries)
#         avg_corrs[f"{size}_bwd"] = np.mean(all_bwd_vs_human_entries)
#         std_corrs[f"{size}_bwd"] = np.std(all_bwd_vs_human_entries)

#     # Plot barplots for each model size and direction
#     fig, ax = plt.subplots(figsize=(5, 3))
#     bar_width = 0.35
#     x = np.arange(len(sizes))
#     ax.bar(
#         x - bar_width/2, [avg_corrs[f"{size}_fwd"] for size in sizes], 
#         bar_width,
#         yerr=[std_corrs[f"{size}_fwd"] for size in sizes],
#         capsize=5,
#         label='Fwd', color='#E8B7D4'
#     )
#     ax.bar(
#         x + bar_width/2, [avg_corrs[f"{size}_bwd"] for size in sizes], 
#         bar_width, yerr=[std_corrs[f"{size}_bwd"] for size in sizes], 
#         capsize=5,
#         label='Bwd', color='#FF7B89'
#     )
#     ax.set_xticks(x)
#     ax.set_xticklabels(["GPT2 (124M)", "GPT2 (355M)", "GPT2 (774M)"])
#     ax.set_ylabel('Correlation')
#     ax.legend()
#     ax.grid(axis='y', linestyle='--', alpha=0.6)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.tight_layout()
#     plt.savefig(f"figs/error_correlation_human_vs_models_{who}_created.pdf")
#     plt.close()

#     # Stats testing comparing all fwd vs human and all bwd vs human
#     t, p = stats.ttest_rel(all_fwd_vs_human, all_bwd_vs_human)
#     print(f"Stats testing fwd-human vs bwd-human")
#     print("-" * 50)
#     print(f"t({len(all_fwd_vs_human)-1}) = {t:.3f}, p = {p:.3f}")

#     # Average and std correlation for all fwd vs human and all bwd vs human
#     avg_corr_fwd = np.mean(all_fwd_vs_human)
#     std_corr_fwd = np.std(all_fwd_vs_human)
#     avg_corr_bwd = np.mean(all_bwd_vs_human)
#     std_corr_bwd = np.std(all_bwd_vs_human)
#     print(f"\nAverage correlation fwd vs human: {avg_corr_fwd:.2f}")
#     print(f"Std correlation fwd vs human: {std_corr_fwd:.2f}")
#     print(f"Average correlation bwd vs human: {avg_corr_bwd:.2f}")
#     print(f"Std correlation bwd vs human: {std_corr_bwd:.2f}")


def plot_combined_correlations(corr_all_models_and_human, llms_flat_names, who):
    plt.rcParams.update({"font.size": 12, "font.weight": "normal"})
    
    # Calculations from plot_model_model_and_model_human_correlation
    avg_model_model_corr = np.nanmean(corr_all_models_and_human[:-1, :-1])
    std_model_model_corr = np.nanstd(corr_all_models_and_human[:-1, :-1])
    avg_model_human_corr = np.nanmean(corr_all_models_and_human[-1, :-1])
    std_model_human_corr = np.nanstd(corr_all_models_and_human[-1, :-1])

    print(f"Average model-model correlation: {avg_model_model_corr:.2f}")
    print(f"Std model-model correlation: {std_model_model_corr:.2f}")
    print(f"Average model-human correlation: {avg_model_human_corr:.2f}")
    print(f"Std model-human correlation: {std_model_human_corr:.2f}")
    print("-" * 50)

    # Calculations from plot_human_vs_model_correlation
    avg_corrs = {}
    std_corrs = {}
    sizes = ["small", "medium", "large"]
    indices = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18)]
    
    all_fwd_vs_human = []
    all_bwd_vs_human = []
    for size, (fwd_idx, bwd_idx) in zip(sizes, zip(indices[::2], indices[1::2])):
        print(f"\n{size} models")
        print("-" * 50)
        print(f"Forward models: {fwd_idx[0]}-{fwd_idx[1]}")
        print(f"Backward models: {bwd_idx[0]}-{bwd_idx[1]}")
        all_fwd_vs_human_entries = corr_all_models_and_human[-1, :-1][fwd_idx[0]:fwd_idx[1]]
        all_bwd_vs_human_entries = corr_all_models_and_human[-1, :-1][bwd_idx[0]:bwd_idx[1]]
        all_fwd_vs_human.extend(all_fwd_vs_human_entries)
        all_bwd_vs_human.extend(all_bwd_vs_human_entries)
        avg_corrs[f"{size}_fwd"] = np.mean(all_fwd_vs_human_entries)
        std_corrs[f"{size}_fwd"] = np.std(all_fwd_vs_human_entries)
        avg_corrs[f"{size}_bwd"] = np.mean(all_bwd_vs_human_entries)
        std_corrs[f"{size}_bwd"] = np.std(all_bwd_vs_human_entries)

    # Statistical testing
    t, p = stats.ttest_rel(all_fwd_vs_human, all_bwd_vs_human)
    print(f"Stats testing fwd-human vs bwd-human")
    print("-" * 50)
    print(f"t({len(all_fwd_vs_human)-1}) = {t:.3f}, p = {p:.3f}")

    # Average and std correlation for all fwd vs human and all bwd vs human
    avg_corr_fwd = np.mean(all_fwd_vs_human)
    std_corr_fwd = np.std(all_fwd_vs_human)
    avg_corr_bwd = np.mean(all_bwd_vs_human)
    std_corr_bwd = np.std(all_bwd_vs_human)
    print(f"\nAverage correlation fwd vs human: {avg_corr_fwd:.2f}")
    print(f"Std correlation fwd vs human: {std_corr_fwd:.2f}")
    print(f"Average correlation bwd vs human: {avg_corr_bwd:.2f}")
    print(f"Std correlation bwd vs human: {std_corr_bwd:.2f}")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

    # Plot 1: Model-Model vs Model-Human correlations (from plot_model_model_and_model_human_correlation)
    bar_width = 0.35
    ax1.bar(
        np.arange(2),
        [avg_model_model_corr, avg_model_human_corr],
        bar_width,
        yerr=[std_model_model_corr, std_model_human_corr],
        capsize=5,
        color=["blue", "orange"],
        tick_label=['Model-Model', 'Model-Human']
    )
    ax1.set_ylabel('Correlation')
    ax1.set_ylim(0, 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Plot 2: Human vs Model correlations by size and direction (from plot_human_vs_model_correlation)
    x = np.arange(len(sizes))
    ax2.bar(
        x - bar_width/2, [avg_corrs[f"{size}_fwd"] for size in sizes], 
        bar_width,
        yerr=[std_corrs[f"{size}_fwd"] for size in sizes],
        capsize=5,
        label='Fwd', color='#E8B7D4'
    )
    ax2.bar(
        x + bar_width/2, [avg_corrs[f"{size}_bwd"] for size in sizes], 
        bar_width, yerr=[std_corrs[f"{size}_bwd"] for size in sizes], 
        capsize=5,
        label='Bwd', color='#FF7B89'
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(["GPT2 (124M)", "GPT2 (355M)", "GPT2 (774M)"])
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"figs/error_correlation_combined_{who}_created.pdf")
    plt.close()


def plot_all_models_and_human_correlation(corr_all_models_and_human, llms_flat_names, who):
    plt.rcParams.update({"font.size": 18, "font.weight": "normal"})

    fig, ax = plt.subplots(figsize=(20, 20))
    cmap = sns.color_palette("inferno", as_cmap=True)
    im = sns.heatmap(corr_all_models_and_human, cmap=cmap, annot=True, ax=ax, cbar=True)

    ticklabels = [f"{llm}" for llm in llms_flat_names] + ["Human experts"]

    xticklabels = ticklabels[:-1]
    yticklabels = ticklabels[1:]
    xticks = np.arange(0.5, len(xticklabels), 1)
    yticks = np.arange(1.5, len(yticklabels)+1, 1)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticklabels(yticklabels, rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    plt.savefig(f"figs/error_correlation_{who}_created_all_llms_heatmap.pdf")


def plot(who, avg_acc_dict_human):
    """
    Human vs individual models and between models correlations, as heatmaps
    """
    corr_all_models_and_human, llms_flat_names = get_all_correlations(who, avg_acc_dict_human)

    # # Plot all model-model and model-human correlations
    # plot_model_model_and_model_human_correlation(corr_all_models_and_human, llms_flat_names, who)

    # # Plot human vs models correlation
    # plot_human_vs_model_correlation(corr_all_models_and_human, who)

    # Plot combined correlations
    plot_combined_correlations(corr_all_models_and_human, llms_flat_names, who)

    # Plot all models and human correlation
    plot_all_models_and_human_correlation(corr_all_models_and_human, llms_flat_names, who)


def main():
    who_created = ["human"]
    for who in who_created:
        avg_acc_dict_human = average_acc_across_cases_participants(who=who)
        plot(who, avg_acc_dict_human)
                

if __name__ == "__main__":
    model_results_dir = "model_results"
    human_results_dir = "human_results"

    # Resources for this analysis
    # Online study data
    online_study_data = pd.read_csv(f"{human_results_dir}/data/participant_data.csv")

    # Remove perm models as not part of brainbench evals.
    llms = {}
    for model_family in model_utils.model_list.keys():
        llms[model_family] = {}
        for llm in model_utils.model_list[model_family].keys():
            if "perm" in llm:
                continue
            llms[model_family][llm] = model_utils.model_list[model_family][llm]
    
    # human 200 cases in model eval order 
    human_abstracts = pd.read_csv("testcases/BrainBench_Human_v0.1.csv")
    human_abstracts_dois = human_abstracts["doi"]

    # machine 100 cases in model eval order
    machine_abstracts = pd.read_csv("testcases/BrainBench_GPT-4_v0.1.csv")
    machine_abstracts_dois = machine_abstracts["doi"]

    # abstract_id (used in online study)  to doi mapper
    mapper = pd.read_csv(f"{human_results_dir}/abstract_id_doi.csv")
    # mapper has one column and 3 ; delimited values `doi;id;abstract_content`
    # we process all rows of mapper into a dict where key is abstract_id and value is doi
    mapper_dict = {}
    for _, row in mapper.iterrows():
        all_stuff = row.iloc[0].split(";")
        doi = all_stuff[0]
        abstract_id = int(float(all_stuff[1]))
        # print(f"doi: {doi}, abstract_id: {abstract_id}")
        mapper_dict[abstract_id] = doi

    # Sanity check 
    # that all doi values in mapper_dict and dois in human_abstracts
    # from column `DOI link (shown below authors' names)` match
    for doi in human_abstracts["doi"]:
        if doi not in mapper_dict.values():
            print(f"DOI {doi} not in mapper_dict")
            break

    main()