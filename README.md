# Beyond Human-Like Processing: Large Language Models Perform Equivalently on Forward and Backward Scientific Text

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="https://github.com/user-attachments/assets/090584ca-12c4-44e4-aa48-d97b510e7fab" style="height: 300px; width: auto; margin-right: 10px;">
    <img src="https://github.com/user-attachments/assets/598edb45-8f9c-419d-95f3-4e2ade596299" style="height: 300px; width: auto;">
</div>



### Work with this repo locally:
```
git clone https://github.com/braingpt-lovelab/backwards --recursive
```
* Will recursively grab the asubmodule for human participant data from this [repo](https://github.com/braingpt-lovelab/brainbench_participant_data/tree/bd1536e99c23d0d1d96d2d3d09bb9a4f52c8e170).
* Will recursively grab the submodule for BrainBench testcases from this [repo](https://github.com/braingpt-lovelab/brainbench_testcases/tree/89869dab3be1ec096dc38931ea33e43268c65d30).

### Repo structure
* `model_training/`: training scripts for both forward and backward GPT-2 models.
* `analyses/`: post-training analyses scripts for producing results in the paper.

### Training
`cd model_training`
* Entry-point is `launch_training.sh` which calls `train.py` or `train_backwards.py` given configurations.
* Training configurations can be set in `configs/` and `accel_config.yaml`.
* Forward and backward tokenizers can be trained from scratch by `tokenizer.py` and `tokenizer_backwards.py`.
* Training data is hosted [here](https://huggingface.co/datasets/BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset/discussions).

### Reproduce analyses from scratch:
`cd analyses`
* Produce model responses: `run_choice.py` and `run_choice_backwards.py`.
* Statistical analyses: `anova_stats.R`.
* Fig. 3: `plot_model_vs_human.py`.
* Fig. 4 & Table 1: `get_ppl_final_val.py` to obtain valiation results and `plot_ppl_val_and_test.py` for plotting.
* Fig. 5: `plot_error_correlation_model_vs_human.py`
* Fig. S1: `neuro_term_tagging.py` to obtain raw results and `python plot_token_analyses.py` for plotting.

### Attribution
```
@article{luo2024beyond,
  title={Beyond Human-Like Processing: Large Language Models Perform Equivalently on Forward and Backward Scientific Text},
  author={Luo, X. and Ramscar, M. and Love, B. C.},
  journal={arXiv preprint arXiv:2411.11061},
  year={2024}
}
```
