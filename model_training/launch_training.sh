#!/bin/bash

# Define the JSON file and expdir
args_json="configs/gpt2_scratch_neuro_tokenizer_bayes_perm.json"

expdir=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(d['outputdir'])")
mkdir -p $expdir

# Extract the JSON contents into Bash-friendly key-value pairs
key_value_pairs=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(' '.join([f'--{key} {value}' for key, value in d.items()]))")

# Launch the Python script with the dynamically created arguments
accelerate launch --config_file accel_config.yaml train_bayes.py $key_value_pairs


# Define the JSON file and expdir
args_json="configs/gpt2-medium_scratch_neuro_tokenizer_bayes_perm.json"

expdir=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(d['outputdir'])")
mkdir -p $expdir

# Extract the JSON contents into Bash-friendly key-value pairs
key_value_pairs=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(' '.join([f'--{key} {value}' for key, value in d.items()]))")

# Launch the Python script with the dynamically created arguments
accelerate launch --config_file accel_config.yaml train_bayes.py $key_value_pairs



# Define the JSON file and expdir
args_json="configs/gpt2-large_scratch_neuro_tokenizer_bayes_perm.json"

expdir=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(d['outputdir'])")
mkdir -p $expdir

# Extract the JSON contents into Bash-friendly key-value pairs
key_value_pairs=$(python3 -c "import json; import sys; d = json.load(open('$args_json')); sys.stdout.write(' '.join([f'--{key} {value}' for key, value in d.items()]))")

# Launch the Python script with the dynamically created arguments
accelerate launch --config_file accel_config.yaml train_bayes.py $key_value_pairs