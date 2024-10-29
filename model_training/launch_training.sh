#!/bin/bash

# Define the JSON file and expdir
args_json="configs/gpt2_scratch_neuro_tokenizer_backwards_seed2.json"

expdir=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(d['outputdir'])")
mkdir -p $expdir

# Extract the JSON contents into Bash-friendly key-value pairs
key_value_pairs=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(' '.join([f'--{key} {value}' for key, value in d.items()]))")

# Launch the Python script with the dynamically created arguments
accelerate launch --config_file accel_config.yaml train_backwards.py $key_value_pairs

# Run another training job with a different seed
args_json="configs/gpt2_scratch_neuro_tokenizer_backwards_seed3.json"
expdir=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(d['outputdir'])")
mkdir -p $expdir
key_value_pairs=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(' '.join([f'--{key} {value}' for key, value in d.items()]))")
accelerate launch --config_file accel_config.yaml train_backwards.py $key_value_pairs