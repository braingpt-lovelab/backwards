import numpy as np
import scipy.stats as stats

comparisons = {
    "between_runs":
        {
            "GPT-2 (124M)": {
                "Fwd": ["gpt2_scratch_neuro_tokenizer_bayes_fwd", "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed2", "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed3"],
                "Bwd": ["gpt2_scratch_neuro_tokenizer_bayes_rev", "gpt2_scratch_neuro_tokenizer_bayes_rev_seed2", "gpt2_scratch_neuro_tokenizer_bayes_rev_seed3"],
                "Perm": ["gpt2_scratch_neuro_tokenizer_bayes_perm", "gpt2_scratch_neuro_tokenizer_bayes_perm_seed2", "gpt2_scratch_neuro_tokenizer_bayes_perm_seed3"]
            },
            "GPT-2 (355M)": {
                "Fwd": ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd", "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed2", "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed3"],
                "Bwd": ["gpt2-medium_scratch_neuro_tokenizer_bayes_rev", "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed2", "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed3"],
                "Perm": ["gpt2-medium_scratch_neuro_tokenizer_bayes_perm", "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed2", "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed3"]
            },
            "GPT-2 (774M)": {
                "Fwd": ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd", "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed2", "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed3"],
                "Bwd": ["gpt2-large_scratch_neuro_tokenizer_bayes_rev", "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed2", "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed3"],
                "Perm": ["gpt2-large_scratch_neuro_tokenizer_bayes_perm", "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed2", "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed3"]
            }
        },
    "between_directions":
        {
            "GPT-2 (124M)": {
                "Fwd vs Bwd": [
                    ["gpt2_scratch_neuro_tokenizer_bayes_fwd", "gpt2_scratch_neuro_tokenizer_bayes_rev"],
                    ["gpt2_scratch_neuro_tokenizer_bayes_fwd_seed2", "gpt2_scratch_neuro_tokenizer_bayes_rev_seed2"],
                    ["gpt2_scratch_neuro_tokenizer_bayes_fwd_seed3", "gpt2_scratch_neuro_tokenizer_bayes_rev_seed3"],
                ],
                "Fwd vs Perm": [
                    ["gpt2_scratch_neuro_tokenizer_bayes_fwd", "gpt2_scratch_neuro_tokenizer_bayes_perm"],
                    ["gpt2_scratch_neuro_tokenizer_bayes_fwd_seed2", "gpt2_scratch_neuro_tokenizer_bayes_perm_seed2"],
                    ["gpt2_scratch_neuro_tokenizer_bayes_fwd_seed3", "gpt2_scratch_neuro_tokenizer_bayes_perm_seed3"],
                ],
                "Bwd vs Perm": [
                    ["gpt2_scratch_neuro_tokenizer_bayes_rev", "gpt2_scratch_neuro_tokenizer_bayes_perm"],
                    ["gpt2_scratch_neuro_tokenizer_bayes_rev_seed2", "gpt2_scratch_neuro_tokenizer_bayes_perm_seed2"],
                    ["gpt2_scratch_neuro_tokenizer_bayes_rev_seed3", "gpt2_scratch_neuro_tokenizer_bayes_perm_seed3"],
                ]
            },
            "GPT-2 (355M)": {
                "Fwd vs Bwd": [
                    ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd", "gpt2-medium_scratch_neuro_tokenizer_bayes_rev"],
                    ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed2", "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed2"],
                    ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed3", "gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed3"],
                ],
                "Fwd vs Perm": [
                    ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd", "gpt2-medium_scratch_neuro_tokenizer_bayes_perm"],
                    ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed2", "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed2"],
                    ["gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed3", "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed3"],
                ],
                "Bwd vs Perm": [
                    ["gpt2-medium_scratch_neuro_tokenizer_bayes_rev", "gpt2-medium_scratch_neuro_tokenizer_bayes_perm"],
                    ["gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed2", "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed2"],
                    ["gpt2-medium_scratch_neuro_tokenizer_bayes_rev_seed3", "gpt2-medium_scratch_neuro_tokenizer_bayes_perm_seed3"],
                ]
            },
            "GPT-2 (774M)": {
                "Fwd vs Bwd": [
                    ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd", "gpt2-large_scratch_neuro_tokenizer_bayes_rev"],
                    ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed2", "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed2"],
                    ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed3", "gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed3"],
                ],
                "Fwd vs Perm": [
                    ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd", "gpt2-large_scratch_neuro_tokenizer_bayes_perm"],
                    ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed2", "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed2"],
                    ["gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed3", "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed3"],
                ],
                "Bwd vs Perm": [
                    ["gpt2-large_scratch_neuro_tokenizer_bayes_rev", "gpt2-large_scratch_neuro_tokenizer_bayes_perm"],
                    ["gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed2", "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed2"],
                    ["gpt2-large_scratch_neuro_tokenizer_bayes_rev_seed3", "gpt2-large_scratch_neuro_tokenizer_bayes_perm_seed3"],
                ]
            }
        }
    }


def compute_stats(model1_ppls, model2_ppls):
    t_stat, p_val = stats.ttest_rel(model1_ppls, model2_ppls)
    pearson_r = np.corrcoef(model1_ppls, model2_ppls)[0, 1]
    cohen_d = (np.mean(model1_ppls) - np.mean(model2_ppls)) / np.sqrt(
        np.var(model1_ppls - model2_ppls, ddof=1)
    )
    pearson_r = np.corrcoef(model1_ppls, model2_ppls)[0, 1]
    return t_stat, p_val, pearson_r, cohen_d


def x_seeds_diffs(comparison, data_type="validation"):
    for model_family, per_direction_models in comparison.items():
        # e.g., "GPT-2 (124M)"
        print(f"\nModel family: {model_family}")

        for direction, model_names in per_direction_models.items():
            # e.g., "Fwd"
            print(f"  Direction: {direction}")

            per_direction_model_pairs_stats = {
                "t_stat": [],
                "p_val": [],
                "pearson_r": [],
                "cohen_d": []
            }
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1_name = model_names[i]
                    model2_name = model_names[j]
                    # print(f"    Comparing [{model1_name}] vs [{model2_name}]")

                    # Load results
                    model1_ppls = np.load(
                        f"model_results/{model1_name}/{type_of_abstract}/all_batches_ppl_{data_type}.npy"
                    )
                    model2_ppls = np.load(
                        f"model_results/{model2_name}/{type_of_abstract}/all_batches_ppl_{data_type}.npy"
                    )

                    # Compute stats
                    t_stat, p_val, pearson_r, cohen_d = compute_stats(model1_ppls, model2_ppls)
                    per_direction_model_pairs_stats["t_stat"].append(t_stat)
                    per_direction_model_pairs_stats["p_val"].append(p_val)
                    per_direction_model_pairs_stats["pearson_r"].append(pearson_r)
                    per_direction_model_pairs_stats["cohen_d"].append(cohen_d)

            # Compute average stats for the direction
            avg_t_stat = np.mean(per_direction_model_pairs_stats["t_stat"])
            avg_p_val = np.mean(per_direction_model_pairs_stats["p_val"])
            avg_pearson_r = np.mean(per_direction_model_pairs_stats["pearson_r"])
            avg_cohen_d = np.mean(per_direction_model_pairs_stats["cohen_d"])
            std_t_stat = np.std(per_direction_model_pairs_stats["t_stat"])
            std_p_val = np.std(per_direction_model_pairs_stats["p_val"])
            std_pearson_r = np.std(per_direction_model_pairs_stats["pearson_r"])
            std_cohen_d = np.std(per_direction_model_pairs_stats["cohen_d"])

            # print for per family and per direction
            print(f"    Average stats for {direction}:")
            print(f"      t_stat: {avg_t_stat:.4f} ± {std_t_stat:.4f}")
            print(f"      p_val: {avg_p_val:.4f} ± {std_p_val:.4f}")
            print(f"      pearson_r: {avg_pearson_r:.4f} ± {std_pearson_r:.4f}")
            print(f"      cohen_d: {avg_cohen_d:.4f} ± {std_cohen_d:.4f}")

                
def x_direction_diffs(comparison, data_type="validation"):
    for model_family, per_direction_models in comparison.items():
        # e.g., "GPT-2 (124M)"
        print(f"\nModel family: {model_family}")


        for direction_pair, model_pairs in per_direction_models.items():
            # e.g., "Fwd vs Bwd"
            print(f"  Direction pair: {direction_pair}")

            per_direction_pair_model_pairs_stats = {
                "t_stat": [],
                "p_val": [],
                "pearson_r": [],
                "cohen_d": []
            }
            for model_pair in model_pairs:
                model1_name, model2_name = model_pair
                # print(f"    Comparing [{model1}] vs [{model2}]")

                # Load results
                model1_ppls = np.load(
                    f"model_results/{model1_name}/{type_of_abstract}/all_batches_ppl_{data_type}.npy"
                )
                model2_ppls = np.load(
                    f"model_results/{model2_name}/{type_of_abstract}/all_batches_ppl_{data_type}.npy"
                )
                
                # Compute stats
                t_stat, p_val, pearson_r, cohen_d = compute_stats(model1_ppls, model2_ppls)
                per_direction_pair_model_pairs_stats["t_stat"].append(t_stat)
                per_direction_pair_model_pairs_stats["p_val"].append(p_val)
                per_direction_pair_model_pairs_stats["pearson_r"].append(pearson_r)
                per_direction_pair_model_pairs_stats["cohen_d"].append(cohen_d)
        
            # Compute average stats for the direction
            avg_t_stat = np.mean(per_direction_pair_model_pairs_stats["t_stat"])
            avg_p_val = np.mean(per_direction_pair_model_pairs_stats["p_val"])
            avg_pearson_r = np.mean(per_direction_pair_model_pairs_stats["pearson_r"])
            avg_cohen_d = np.mean(per_direction_pair_model_pairs_stats["cohen_d"])
            std_t_stat = np.std(per_direction_pair_model_pairs_stats["t_stat"])
            std_p_val = np.std(per_direction_pair_model_pairs_stats["p_val"])
            std_pearson_r = np.std(per_direction_pair_model_pairs_stats["pearson_r"])
            std_cohen_d = np.std(per_direction_pair_model_pairs_stats["cohen_d"])

            # print for per family and per direction pair
            print(f"    Average stats for {direction_pair}:")
            print(f"      t_stat: {avg_t_stat:.4f} ± {std_t_stat:.4f}")
            print(f"      p_val: {avg_p_val:.4f} ± {std_p_val:.4f}")
            print(f"      pearson_r: {avg_pearson_r:.4f} ± {std_pearson_r:.4f}")
            print(f"      cohen_d: {avg_cohen_d:.4f} ± {std_cohen_d:.4f}")
           
        
def main():
    x_seeds_diffs(comparisons["between_runs"], data_type="validation")
    x_direction_diffs(comparisons["between_directions"], data_type="validation")


if __name__ == "__main__":
    results_dir = "model_results"
    type_of_abstract = "human_abstracts"

    main()
