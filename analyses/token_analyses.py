import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib.gridspec import GridSpec


def _load_tagging_results(fname):
    """
    Load and convert to a dictionary,
    each entry is a row, key is the word, value is the result.
    """
    tagging_map = {}
    with open(f"{results_dir}/{fname}", 'r') as f:
        lines = f.readlines()
        for line in lines:
            word, result = line.strip().split(", ")
            if result == "yes":
                result = 1
            else:
                result = 0
            tagging_map[word] = result
    return tagging_map


def _get_proportion(tagging_map, vocab, direction="forwards"):
    n_neuro_terms = 0
    for token in vocab:
        if direction == "backwards":
            if token.startswith('Ġ'):
                token = 'Ġ' + token[1:][::-1]
            else:
                token = token[::-1]

        token = token.replace('Ġ', '').lower()
        if token in tagging_map and tagging_map[token] == 1:
            n_neuro_terms += 1
    return n_neuro_terms / len(vocab)


def common_term_proportion():
    """
    Check shared tokens in both forwards and backwards (reversed) vocabs.
    """
    # Replace Ġ with space from each key if there is one.
    forwards_filtered_keys = set()
    print(f"Number of forwards keys: {len(forwards_vocab.keys())}")
    for k in forwards_vocab.keys():
        forwards_filtered_keys.add(k)

    backwards_filtered_keys = set()
    print(f"Number of backwards keys: {len(backwards_vocab.keys())}")
    for k in backwards_vocab.keys():
        if k.startswith('Ġ'):
            k = 'Ġ' + k[1:][::-1]
        else:
            k = k[::-1]
        backwards_filtered_keys.add(k)

    # Save as txt, common and unique keys in filtered version.
    forwards_filtered_keys = set(forwards_filtered_keys)
    backwards_filtered_keys = set(backwards_filtered_keys)
    filtered_common_keys = forwards_filtered_keys.intersection(backwards_filtered_keys)
    print(f"Number of common keys: {len(filtered_common_keys)}")

    return len(filtered_common_keys) / len(forwards_vocab)
    


def neuro_term_proportion():
    """
    Compare proportion of tokens (deemed by GPT) to be a common
    term in Neuroscience.
    """
    forwards_tagging_map = _load_tagging_results(forwards_neuro_term_fname)
    backwards_tagging_map = _load_tagging_results(backwards_neuro_term_fname)

    forwards_proportion = _get_proportion(
        forwards_tagging_map, forwards_vocab, direction="forwards"
    )
    backwards_proportion = _get_proportion(
        backwards_tagging_map, backwards_vocab, direction="backwards"
    )

    print(f"forwards neuro term proportion: {forwards_proportion:.2f}")
    print(f"backwards neuro term proportion: {backwards_proportion:.2f}")
    return forwards_proportion, backwards_proportion


def main():
    common_proportion = common_term_proportion()
    forwards_proportion, backwards_proportion = neuro_term_proportion()

    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

    # Create figure and define grid layout
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 2, figure=fig)

    # Top plot - Venn Diagram
    ax1 = fig.add_subplot(gs[0, :])  # Span both columns at the top
    total = 1.0
    shared = common_proportion * total
    exclusive = (total - shared) / 2

    venn_diagram = venn2(
        subsets=(exclusive, exclusive, shared), 
        # set_labels=('forwards\nVocab.', 'Neuro Tokenizer\nVocab.'),
        set_labels=('', ''),
        ax=ax1
    )
    
    venn_diagram.get_patch_by_id('10').set_alpha(1)
    venn_diagram.get_patch_by_id('01').set_alpha(1)
    venn_diagram.get_patch_by_id('11').set_alpha(0.5)
    venn_diagram.get_patch_by_id('10').set_color('skyblue')
    venn_diagram.get_patch_by_id('01').set_color('lightgreen')
    venn_diagram.get_patch_by_id('11').set_color('grey')
    venn_diagram.get_label_by_id('10').set_text('')
    venn_diagram.get_label_by_id('01').set_text('')
    venn_diagram.get_label_by_id('11').set_text(f'Shared\nTokens\n{shared*100:.1f}%')

    # Add label A to upper left corner of ax1
    ax1.text(-0.8, 0.5, 'A', fontsize=16, fontweight='bold')

    # Bottom left - Forwards Vocab Pie Chart
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.pie([forwards_proportion, 1 - forwards_proportion], labels=["Neuro\nTokens", ""],
            autopct='%1.1f%%', startangle=90, explode=(0.1, 0.), colors=['cyan', 'skyblue'])
    ax2.set_xlabel("Forwards\nVocab.", fontsize=16, fontweight='bold')
    ax2.text(-2, 1.5, 'B', fontsize=16, fontweight='bold')

    # Bottom right - Backwards Vocab Pie Chart
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.pie([backwards_proportion, 1 - backwards_proportion], labels=["Neuro\nTokens", ""],
            autopct='%1.1f%%', startangle=90, explode=(0.1, 0.), colors=['cyan', 'lightgreen'])
    ax3.set_xlabel("Backwards\nVocab.", fontsize=16, fontweight='bold')
    ax3.text(-2, 1.5, 'C', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig("figs/shared_terms_and_neuro_term_proportion.pdf")
    

if __name__ == "__main__":
    results_dir = "token_results"
    data_dir = "data"
    forwards_vocab = json.load(open(f'{data_dir}/vocab_neuro.json'))
    backwards_vocab = json.load(open(f'{data_dir}/vocab_neuro_backwards.json'))

    forwards_neuro_term_fname = f"forwards_filtered__neuro_term_tagging_results.txt"
    backwards_neuro_term_fname = f"backwards_filtered__neuro_term_tagging_results.txt"

    main()