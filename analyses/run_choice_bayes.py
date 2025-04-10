import os
import argparse

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F 

from utils import data_utils
from utils import model_utils
from utils import general_utils


def forward_pass(model, tokenizer, choices):
    """
    Args:
        - choices (list): list of strings, where each string is a prompt

    Perform a single forward pass over a testcase 
    (i.e., a prompt with choices) and computes perplexities
    for each choice.
    """
    # Forward pass to get nll and convert to ppl
    ppl = []
    for choice_index, prompt in enumerate(choices):
        with torch.no_grad():
            prompt = tokenizer(prompt, return_tensors='pt').to("cuda")

            # Apply bayes-unique data processing
            # 1) if `rev` in model name, reverse the token ids
            # 2) always add bos_token_id to the beginning of the input_ids
            if "rev" in llm:
                prompt["input_ids"] = torch.flip(prompt["input_ids"], dims=[1])
            bos_token_id = tokenizer.bos_token_id
            prompt["input_ids"] = torch.cat(
                [torch.tensor([[bos_token_id]]).to("cuda"), prompt["input_ids"]], dim=1
            )

            if "token_type_ids" in prompt:
                prompt.pop("token_type_ids")

            output = model(
                input_ids=prompt["input_ids"], 
                labels=prompt["input_ids"]
            )

            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_labels = prompt["input_ids"][..., 1:].contiguous()

            # Compute loss
            # - flatten logits and labels
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # - logsoftmax must ignore the BOS column
            mask = torch.zeros_like(shift_logits)
            bos_token_id = tokenizer.bos_token_id
            mask[:, bos_token_id] = float('-inf')  # Set the BOS column logits to -inf
            masked_shift_logits = shift_logits + mask
            
            # - Softmax and compute loss
            log_probs = torch.nn.functional.log_softmax(masked_shift_logits, dim=-1)
            true_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1).float()
            nll = -true_log_probs.mean()
            ppl.append(np.exp(nll.item()))
    return ppl


@general_utils.timer
def main(llm, abstracts_fpath):
    np.random.seed(42)

    # Load model, tokenizer
    model, tokenizer = model_utils.load_model_and_tokenizer(llm)

    # Load dataset
    df = pd.read_csv(abstracts_fpath)
    prompt_template = data_utils.read_prompt_template(llm)

    PPL_A_and_B = []
    true_labels = []
    for abstract_index, abstract in enumerate(df["combined_abstract"]):
        original_abstract, incorrect_abstract = data_utils.extract_abstract_pair(abstract)

        # Randomly shuffle to determine which abstract is A and which is B,
        # keep a record of the correct choice, which is used to determine
        # later if the model's choice is correct
        if np.random.rand() > 0.5:
            original_abstract, incorrect_abstract = incorrect_abstract, original_abstract
            choice_true = "B"
        else:
            choice_true = "A"

        # choices is [prompt_A, prompt_B]
        # where each prompt is the question + one of the abstracts as option.
        choices = data_utils.prepare_prompt_multiple_choice_harness(
            original_abstract, incorrect_abstract, prompt_template, 
        )

        print(
            f"-"*70 + "\n",
            f"*** Abstract index: {abstract_index} ***",
        )

        # Forward each prompt to get nll and convert to ppl
        ppl = forward_pass(model, tokenizer, choices)
        PPL_A_and_B.append(ppl)
        true_labels.append(0 if choice_true == "A" else 1)

    PPL_A_and_B = np.array(PPL_A_and_B)
    true_labels = np.array(true_labels)

    # Compute accuracy
    tie_indices = []
    pred_labels = np.ones(PPL_A_and_B.shape[0], dtype=np.int32)
    for i, (ppl_A, ppl_B) in enumerate(PPL_A_and_B):
        if ppl_A < ppl_B:
            pred_labels[i] = 0
        elif ppl_A > ppl_B:
            pred_labels[i] = 1
        else:
            pred_labels[i] = -1
            tie_indices.append(i)
    
    print(f"Number of ties: {len(tie_indices)}")

    # Accuracy after removing ties
    acc = np.sum(pred_labels == true_labels) / (PPL_A_and_B.shape[0])
    print(f"Accuracy: {acc}")

    np.save(f"{results_dir}/PPL_A_and_B.npy", PPL_A_and_B)
    np.save(f"{results_dir}/labels.npy", true_labels)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_human_abstract", type=str, default="True")

    if parser.parse_args().use_human_abstract == "True":
        use_human_abstract = True
    else:
        use_human_abstract = False

    llms = [
        # "gpt2_scratch_neuro_tokenizer_bayes_fwd",
        # "gpt2_scratch_neuro_tokenizer_bayes_rev",
        "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed2",
        "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed3",
    ]

    for llm in llms:
        if use_human_abstract:
            type_of_abstract = 'human_abstracts'
            abstracts_fpath = "testcases/BrainBench_Human_v0.1.csv"
        else:
            type_of_abstract = 'llm_abstracts'
            abstracts_fpath = "testcases/BrainBench_GPT-4_v0.1.csv"
        results_dir = f"model_results/{llm.replace('/', '--')}/{type_of_abstract}"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        main(llm, abstracts_fpath)
    
