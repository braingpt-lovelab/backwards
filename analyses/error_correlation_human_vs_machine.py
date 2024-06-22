import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils import model_utils

plt.rcParams.update({"font.size": 22, "font.weight": "bold"})


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


def plot(who, avg_acc_dict_human):
    """
    Human vs individual models and between models correlations, as heatmap
    """
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
            # skip upper triangle including diagonal
            if j >= i:
                continue
            elif i == len(corr_all_models_and_human) - 1:
                # last row
                avg_acc_dict_machine_1 = ppl_diffs_for_single_model(llms_flat[j], who=who)
                avg_acc_dict_machine_2 = avg_acc_dict_human
                corr, _, _, _, _ = per_model_model_error_corr(
                    avg_acc_dict_machine_1, 
                    avg_acc_dict_machine_2
                )
                corr_all_models_and_human[i][j] = corr
            elif j == len(corr_all_models_and_human) - 1:
                # last column
                avg_acc_dict_machine_1 = avg_acc_dict_human
                avg_acc_dict_machine_2 = ppl_diffs_for_single_model(llms_flat[i], who=who)
                corr, _, _, _, _ = per_model_model_error_corr(
                    avg_acc_dict_machine_1, 
                    avg_acc_dict_machine_2
                )
                corr_all_models_and_human[i][j] = corr
            else:
                avg_acc_dict_machine_1 = ppl_diffs_for_single_model(llms_flat[i], who=who)
                avg_acc_dict_machine_2 = ppl_diffs_for_single_model(llms_flat[j], who=who)
                corr, _, _, _, _ = per_model_model_error_corr(
                    avg_acc_dict_machine_1, 
                    avg_acc_dict_machine_2
                )
                corr_all_models_and_human[i][j] = corr
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_all_models_and_human, dtype=bool), k=0)

    # Set the upper triangle values to a specific color (white) or make them transparent
    corr_all_models_and_human[mask] = np.nan  # Set upper triangle values to NaN

    # Compute average and std correlation of LLM vs LLM values
    avg_corr = np.nanmean(corr_all_models_and_human[:-1, :-1])
    std_corr = np.nanstd(corr_all_models_and_human[:-1, :-1])
    print(f"Average correlation: {avg_corr:.2f}")
    print(f"Std correlation: {std_corr:.2f}")

    # Compute average and std correlation of LLM vs human values
    avg_corr = np.nanmean(corr_all_models_and_human[-1, :-1])
    std_corr = np.nanstd(corr_all_models_and_human[-1, :-1])
    print(f"Average correlation: {avg_corr:.2f}")
    print(f"Std correlation: {std_corr:.2f}")

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

    llms = model_utils.model_list

    # We only consider GPT-2 family models
    llms = {k: v for k, v in llms.items() if "gpt2" in k}

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