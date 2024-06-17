import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def ppl_distributional_diff():
    print("Forwards")
    forwards_path = f"{forwards_dir}/PPL_A_and_B.npy"
    forwards_PPL_A_and_B = np.load(forwards_path)
    forwards_PPL_diff = forwards_PPL_A_and_B[:, 0] - forwards_PPL_A_and_B[:, 1]
    print(f"mean: {np.mean(forwards_PPL_diff)}, std: {np.std(forwards_PPL_diff)}")

    print("\nBackwards")
    backwards_path = f"{backwards_dir}/PPL_A_and_B.npy"
    backwards_PPL_A_and_B = np.load(backwards_path)
    backwards_PPL_diff = backwards_PPL_A_and_B[:, 0] - backwards_PPL_A_and_B[:, 1]
    print(f"mean: {np.mean(backwards_PPL_diff)}, std: {np.std(backwards_PPL_diff)}")

    fig, ax = plt.subplots()
    sns.kdeplot(forwards_PPL_diff, ax=ax, label="Forwards", color="blue")
    sns.kdeplot(backwards_PPL_diff, ax=ax, label="Backwards", color="orange")
    ax.hist(forwards_PPL_diff, bins=20, alpha=0.2, color="blue", density=True)
    ax.hist(backwards_PPL_diff, bins=20, alpha=0.2, color="orange", density=True)
    ax.legend()

    # Significance test
    t_stat, p_val = stats.ttest_ind(forwards_PPL_diff, backwards_PPL_diff)
    print(f"t={t_stat}, p={p_val}")
    plt.title("PPL Distributional Difference Forwards")
    plt.savefig("PPL_distributional_diff_forwards.png")


if __name__ == "__main__":
    forwards_dir = "model_results/gpt2_scratch_neuro_tokenizer/human_abstracts"
    backwards_dir = "model_results/gpt2_scratch_neuro_tokenizer_backwards/human_abstracts"
    # forwards_dir = "model_results/gpt2-large_scratch_neuro_tokenizer/human_abstracts"
    # backwards_dir = "model_results/gpt2-large_scratch_neuro_tokenizer_backwards/human_abstracts"

    ppl_distributional_diff()
