# Probability Consistency in Large Language Models: Theoretical Foundations Meet Empirical Discrepancies

### Work with this repo locally:
```
git clone https://github.com/braingpt-lovelab/backwards --recursive
```
* Will recursively grab the asubmodule for human participant data from this [repo](https://github.com/braingpt-lovelab/brainbench_participant_data/tree/bd1536e99c23d0d1d96d2d3d09bb9a4f52c8e170).
* Will recursively grab the submodule for BrainBench testcases from this [repo](https://github.com/braingpt-lovelab/brainbench_testcases/tree/89869dab3be1ec096dc38931ea33e43268c65d30).

### Model weights
All model weights (including checkpoints) are hosted [here](https://huggingface.co/llm-probability).

### Repo structure
* `model_training/`: training scripts for forward, backward and permuted GPT-2 models.
* `analyses/`: post-training analyses scripts for producing results in the paper.

### Training
`cd model_training`
* Entry-point is `launch_training.sh` which calls `train_bayes.py` given configurations.
* Training configurations can be set in `configs/` and `accel_config.yaml`.
* Forward tokenizer can be trained from scratch by `tokenizer.py`.
* Training data is hosted [here](https://huggingface.co/datasets/BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset/discussions).

### Reproduce analyses from scratch:
`cd analyses`
* Fig. 1, S.1: `plot_x_models_train_val_loss_diffs.py`
* Fig. 2, 3, S.6 - S.21: `plot_attn_weights_by_distance.py`
* Fig. 4, S.22, S.23: `get_hidden_states_for_rsa.py` to save hidden states on disk and `plot_hidden_states_rsa.py` for plotting
* Tab. 2, S.1, S.2: `plot_x_models_x_seeds_x_directions_diffs_bayes.py`
* Fig. S.1: `plot_x_models_train_val_losses.py`
* Fig. S.3, S.4, S.5: `plot_attn_weights_by_distance_pretrained.py`
* Fig. S.24: `run_choice_bayes.py` to obtain model responses to BrainBench and `plot_model_vs_human_x_seeds.py` for plotting
* Fig. S.25, S.26: `plot_error_correlation_model_vs_human.py`
* Statistical analyses: `anova_stats.R`.

### Relation to [arXiv:2411.11061](https://arxiv.org/abs/2411.11061)
This paper is a major follow-up to [*Beyond Human-Like Processing: Large Language Models Perform Equivalently on Forward and Backward Scientific Text*](https://arxiv.org/abs/2411.11061). It corrects methodological issues from the earlier work and includes extensive additional training and analysis informed by newly established theoretical proofs.

For the latest developments, we recommend reading this paper. To access the code associated with the previous work, use:
```
git checkout arxiv.org/abs/2411.11061
```
  
### Attribution
```
@misc{luo2025probabilityconsistencylargelanguage,
      title={Probability Consistency in Large Language Models: Theoretical Foundations Meet Empirical Discrepancies}, 
      author={Xiaoliang Luo and Xinyi Xu and Michael Ramscar and Bradley C. Love},
      year={2025},
      eprint={2505.08739},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.08739}, 
}
```
