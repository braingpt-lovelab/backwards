import os
import json
from datasets import load_dataset

data_path = "BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset"
cache_dir = "cache"
dataset = load_dataset(data_path=data_path, cache_dir=cache_dir)


def get_training_corpus(dataset):
    dataset = dataset["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        # For each sample, reverse the text
        samples["text"] = [sample[::-1] for sample in samples["text"]]
        yield samples["text"]

training_corpus = get_training_corpus(dataset)
n_examples = len(next(training_corpus))
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
example = next(training_corpus)[:1][:5][0]
print(example)
print(old_tokenizer.tokenize(example))

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, old_tokenizer.vocab_size)
print(tokenizer.tokenize(example))
print(tokenizer.vocab_size)

# Save the new tokenizer
if not os.path.exists("cache"):
    os.makedirs("cache")
tokenizer.save_pretrained("cache/gpt2_neuro_tokenizer_backwards")