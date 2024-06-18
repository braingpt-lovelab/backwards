import transformers
import torch


# llms = [
#         "gpt2_scratch_neuro_tokenizer",
#         "finetune_gpt2",
#         "gpt2",
#         "gpt2_init",
#         "gpt2_scratch"
#     ]

model_list = {
    "gpt2": {
        "gpt2_init": {
            "llm": "GPT-2 (untrained)",
            "color": '#D2A9B0',
            "alpha": 0.2,
            "hatch": "",
        },
        "gpt2": {
            "llm": "GPT-2 (pretrained)",
            "color": '#D2A9B0',
            "alpha": 0.5,
            "hatch": "",
        },
        "gpt2_scratch_neuro_tokenizer": {
            "llm": "GPT-2 (scratch)",
            "color": '#D2A9B0',
            "alpha": 0.7,
            "hatch": "*",
        },
        "gpt2_scratch_neuro_tokenizer_backwards": {
            "llm": "Reversed GPT-2 (scratch)",
            "color": '#D2A9B0',
            "alpha": 0.9,
            "hatch": "/",
        },
        "gpt2-large_init": {
            "llm": "GPT-2 Large (untrained)",
            "color": '#758EB7',
            "alpha": 0.2,
            "hatch": "",
        },
        "gpt2-large": {
            "llm": "GPT-2 Large (pretrained)",
            "color": '#758EB7',
            "alpha": 0.4,
            "hatch": "",
        },
        "gpt2-large_scratch_neuro_tokenizer": {
            "llm": "GPT-2 Large (scratch)",
            "color": '#758EB7',
            "alpha": 0.9,
            "hatch": "*",
        },
        "gpt2-large_scratch_neuro_tokenizer_backwards": {
            "llm": "Reversed GPT-2 Large (scratch)",
            "color": '#758EB7',
            "alpha": 0.9,
            "hatch": "/",
        },
    },
    # "facebook": {
    #     "facebook/galactica-6.7b": {
    #         "llm": "Galactica-6.7B",
    #         "color": '#A5CAD2',
    #         "alpha": 0.3,
    #         "hatch": "/",
    #         "n_params": 7,
    #         "Initial_Release": "Nov\n2022",
    #     },
    #     "facebook/galactica-30b": {
    #         "llm": "Galactica-30B",
    #         "color": '#758EB7',
    #         "alpha": 0.5,
    #         "hatch": "/",
    #         "n_params": 30,
    #         "Initial_Release": "Nov\n2022",
    #     },
    #     "facebook/galactica-120b": {
    #         "llm": "Galactica-120B",
    #         "color": '#6F5F90',
    #         "alpha": 0.9,
    #         "hatch": "/",
    #         "n_params": 120,
    #         "Initial_Release": "Nov\n2022",
    #     },
    # },
    # "falcon": {
    #     "tiiuae/falcon-40b": {
    #         "llm": "Falcon-40B",
    #         "color": '#E1C0D8',
    #         "alpha": 0.5,
    #         "hatch": None,
    #         "n_params": 40,
    #         "Initial_Release": "May\n2023",
    #     },
    #     "tiiuae/falcon-40b-instruct": {
    #         "llm": "Falcon-40B (instruct)",
    #         "color": '#E1C0D8',
    #         "alpha": 0.5,
    #         "hatch": "/",
    #         "n_params": 40,
    #         "Initial_Release": "May\n2023",
    #     },
    #     "tiiuae/falcon-180B": {
    #         "llm": "Falcon-180B",
    #         "color": '#D2A9B0',
    #         "alpha": 0.9,
    #         "hatch": None,
    #         "n_params": 180,
    #         "Initial_Release": "May\n2023",
    #     },
    #     "tiiuae/falcon-180B-chat": {
    #         "llm": "Falcon-180B (chat)",
    #         "color": '#D2A9B0',
    #         "alpha": 0.9,
    #         "hatch": "/",
    #         "n_params": 180,
    #         "Initial_Release": "May\n2023",
    #     },
    # },
    # "llama": {
    #     "meta-llama/Llama-2-7b-hf": {
    #         "llm": "Llama-2-7B",
    #         "color": '#D1DCE2',
    #         "alpha": 0.3,
    #         "hatch": None,
    #         "n_params": 7,
    #         "Initial_Release": "July\n2023",
    #     },
    #     "meta-llama/Llama-2-7b-chat-hf": {
    #         "llm": "Llama-2-7B (chat)",
    #         "color": '#D1DCE2',
    #         "alpha": 0.3,
    #         "hatch": "/",
    #         "n_params": 7,
    #         "Initial_Release": "July\n2023",
    #     },
    #     "meta-llama/Llama-2-13b-hf": {
    #         "llm": "Llama-2-13B",
    #         "color": '#B3DDD1',
    #         "alpha": 0.5,
    #         "hatch": None,
    #         "n_params": 13,
    #         "Initial_Release": "July\n2023",
    #     },
    #     "meta-llama/Llama-2-13b-chat-hf": {
    #         "llm": "Llama-2-13B (chat)",
    #         "color": '#B3DDD1',
    #         "alpha": 0.5,
    #         "hatch": "/",
    #         "n_params": 13,
    #         "Initial_Release": "July\n2023",
    #     },
    #     "meta-llama/Llama-2-70b-hf": {
    #         "llm": "Llama-2-70B",
    #         "color": '#80BEAF',
    #         "alpha": 0.9,
    #         "hatch": None,
    #         "n_params": 70,
    #         "Initial_Release": "July\n2023",
    #     },
    #     "meta-llama/Llama-2-70b-chat-hf": {
    #         "llm": "Llama-2-70B (chat)",
    #         "color": '#80BEAF',
    #         "alpha": 0.9,
    #         "hatch": "/",
    #         "n_params": 70,
    #         "Initial_Release": "July\n2023",
    #     },
    # },
    # "mistralai": {
    #     "mistralai/Mistral-7B-v0.1": {
    #         "llm": "Mistral-7B",
    #         "color": '#FA9284',
    #         "alpha": 1,
    #         "hatch": None,
    #         "n_params": 7,
    #         "Initial_Release": "Sept\n2023",
    #     },
    #     "mistralai/Mistral-7B-Instruct-v0.1": {
    #         "llm": "Mistral-7B (instruct)",
    #         "color": '#FA9284',
    #         "alpha": 1,
    #         "hatch": "/",
    #         "n_params": 7,
    #         "Initial_Release": "Sept\n2023",
    #     },
    # },
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
            "gpt2_scratch",
            "finetune_gpt2",
            "finetune_gpt2_lr2e-6",
            "gpt2_scratch_neuro_tokenizer",
            "gpt2-large_scratch_neuro_tokenizer",
            "gpt2_scratch_neuro_tokenizer_backwards",
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
    
    # Load model untrained (config only)
    # elif model_fpath == "gpt2_init":
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