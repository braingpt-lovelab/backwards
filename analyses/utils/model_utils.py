import transformers
import torch

model_list = {
    "gpt2": {
        "gpt2_scratch_neuro_tokenizer": {
            "llm": "GPT2-124M",
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2_scratch_neuro_tokenizer_backwards": {
            "llm": "GPT2-124M (backward)",
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "\\",
        },
        "gpt2_scratch_neuro_tokenizer_fwdModel_bwdText": {
            "llm": "GPT2-124M (fmbt)",
            "color": '#758EB7',
            "alpha": 0.8,
            "hatch": "/",
        },
    },
    "gpt2-medium": {
        "gpt2-medium_scratch_neuro_tokenizer": {
            "llm": "GPT2-355M",
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2-medium_scratch_neuro_tokenizer_backwards": {
            "llm": "GPT2-355M (backward)",
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "\\",
        },
        "gpt2-medium_scratch_neuro_tokenizer_fwdModel_bwdText": {
            "llm": "GPT2-355M (fmbt)",
            "color": '#6F5F90',
            "alpha": 0.8,
            "hatch": "/",
        },
    },
    "gpt2-large": {
        "gpt2-large_scratch_neuro_tokenizer": {
            "llm": "GPT2-774M",
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "",
        },
        "gpt2-large_scratch_neuro_tokenizer_backwards": {
            "llm": "GPT2-774M (backward)",
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "\\",
        },
        "gpt2-large_scratch_neuro_tokenizer_fwdModel_bwdText": {
            "llm": "GPT2-774M (fmbt)",
            "color": '#8A5082',
            "alpha": 0.8,
            "hatch": "/",
        },
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
            "gpt2_scratch_neuro_tokenizer_fwdModel_bwdText",
            "gpt2-medium_scratch_neuro_tokenizer",
            "gpt2-medium_scratch_neuro_tokenizer_backwards",
            "gpt2-medium_scratch_neuro_tokenizer_fwdModel_bwdText",
            "gpt2-large_scratch_neuro_tokenizer",
            "gpt2-large_scratch_neuro_tokenizer_backwards",
            "gpt2-large_scratch_neuro_tokenizer_fwdModel_bwdText",
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