import transformers
import torch

model_list_noseeds = {
    "gpt2": {
        "gpt2_scratch_neuro_tokenizer_bayes_fwd": {
            "llm": "GPT2-124M (fwd)",
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2_scratch_neuro_tokenizer_bayes_rev": {
            "llm": "GPT2-124M (rev)",
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "\\",
        }
    },
    "gpt2-medium": {
        "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd": {
            "llm": "GPT2-355M (fwd)",
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2-medium_scratch_neuro_tokenizer_bayes_rev": {
            "llm": "GPT2-355M (rev)",
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "\\",
        }
    },
    "gpt2-large": {
        "gpt2-large_scratch_neuro_tokenizer_bayes_fwd": {
            "llm": "GPT2-774M (fwd)",
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2-large_scratch_neuro_tokenizer_bayes_rev": {
            "llm": "GPT2-774M (rev)",
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "\\",
        }
    }
}

model_list = {
    "gpt2": {
        "gpt2_scratch_neuro_tokenizer_bayes_fwd": {
            "llm": "GPT2-124M (fwd seed1)",
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed2": {
            "llm": "GPT2-124M (fwd seed2)",
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "..",
        },
        "gpt2_scratch_neuro_tokenizer_bayes_fwd_seed3": {
            "llm": "GPT2-124M (fwd seed3)",
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "oo",
        },
        "gpt2_scratch_neuro_tokenizer_bayes_rev": {
            "llm": "GPT2-124M (rev seed1)",
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "\\",
        }
    },
    "gpt2-medium": {
        "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd": {
            "llm": "GPT2-355M (fwd seed1)",
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed2": {
            "llm": "GPT2-355M (fwd seed2)",
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "..",
        },
        "gpt2-medium_scratch_neuro_tokenizer_bayes_fwd_seed3": {
            "llm": "GPT2-355M (fwd seed3)",
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "oo",
        },
        "gpt2-medium_scratch_neuro_tokenizer_bayes_rev": {
            "llm": "GPT2-355M (rev seed1)",
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "\\",
        }
    },
    "gpt2-large": {
        "gpt2-large_scratch_neuro_tokenizer_bayes_fwd": {
            "llm": "GPT2-774M (fwd seed1)",
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed2": {
            "llm": "GPT2-774M (fwd seed2)",
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "..",
        },
        "gpt2-large_scratch_neuro_tokenizer_bayes_fwd_seed3": {
            "llm": "GPT2-774M (fwd seed3)",
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "oo",
        },
        "gpt2-large_scratch_neuro_tokenizer_bayes_rev": {
            "llm": "GPT2-774M (rev seed1)",
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "\\",
        }
    }
}

def load_model_and_tokenizer(model_fpath, tokenizer_only=False):
    if tokenizer_only:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_fpath,
        )
        return tokenizer
    
    load_in_8bit = False
    torch_dtype = torch.float16


    # Load model trained from scratch from local checkpoint
    if model_fpath in [
            "gpt2_scratch_neuro_tokenizer",
            "gpt2_scratch_neuro_tokenizer_backwards",
            "gpt2-medium_scratch_neuro_tokenizer",
            "gpt2-medium_scratch_neuro_tokenizer_backwards",
            "gpt2-large_scratch_neuro_tokenizer",
            "gpt2-large_scratch_neuro_tokenizer_backwards",
        ]:
        model_fpath = f"/home/ken/projects/matching_experts/model_training/exp/{model_fpath}/checkpoint.4"
        print("Loading GPT2 model from", model_fpath)
        model = transformers.GPT2LMHeadModel.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            model_fpath,    
        )
    
    # For the same models above but trained with different seeds
    elif "seed" in model_fpath:
        model_fpath = f"/home/ken/projects/backwards/model_training/exp/{model_fpath}/checkpoint.4"
        print("Loading GPT2 model from", model_fpath)
        model = transformers.GPT2LMHeadModel.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            model_fpath,    
        )
    
    # Load model untrained (config only)
    elif "init" in model_fpath:
        model_name = model_fpath.split("_")[0]
        print(f"Loading {model_name} model untrained")
        from transformers import AutoConfig, AutoModelForCausalLM
        model_config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(model_config).to('cuda')
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)

    # Load bayes fwd/rev models
    elif "bayes" in model_fpath:
        model_fpath = f"/home/ken/projects/backwards/model_training/exp/{model_fpath}/checkpoint.4"
        print("Loading GPT2 model from", model_fpath)
        model = transformers.GPT2LMHeadModel.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            model_fpath,    
        )

    # Load pretrained model
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_fpath,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_fpath,
        )

    return model, tokenizer