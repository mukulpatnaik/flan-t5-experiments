import pandas as pd
import torch
import argparse
import os

# set the environment variable HUGGINGFACE_HUB_CACHE to the path of the cache directory
os.environ["HUGGINGFACE_HUB_CACHE"] = "/projects/mupa3718/flan-t5-experiments/.cache"
# set the environment variable TRANSFORMERS_CACHE to the path of the cache directory
os.environ["TRANSFORMERS_CACHE"] = "/projects/mupa3718/flan-t5-experiments/.cache"
# set the environment variable HF_HOME. to the path of the cache directory
os.environ["HF_HOME"] = "/projects/mupa3718/flan-t5-experiments/.cache"
# set the environment variable XDG_CACHE_HOME+ /huggingface to the path of the cache directory
os.environ["XDG_CACHE_HOME"] = "/projects/mupa3718/flan-t5-experiments/.cache"
# set the environment variable HF_DATASETS_CACHE to the path of the cache directory
os.environ["HF_DATASETS_CACHE"] = "/projects/mupa3718/flan-t5-experiments/.cache"
# set the environment variable HF_MODEL_CACHE to the path of the cache directory
os.environ["HF_MODEL_CACHE"] = "/projects/mupa3718/flan-t5-experiments/.cache"
# set the environment variable HF_CONFIG_CACHE to the path of the cache directory
os.environ["HF_CONFIG_CACHE"] = "/projects/mupa3718/flan-t5-experiments/.cache"
# set the environment variable HF_TF_CACHE to the path of the cache directory
os.environ["HF_TF_CACHE"] = "/projects/mupa3718/flan-t5-experiments/.cache"

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from accelerate import Accelerator

def get_model_and_tokenizer(model_size):
    if model_size in ["small", "large", "base", "xl", "xxl"]:
        model = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{model_size}", device_map="auto")
        # model.to("mps")
        tokenizer = T5Tokenizer.from_pretrained(f"google/flan-t5-{model_size}")
    elif model_size == "eightbitmodel":
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", load_in_8bit=True)
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    elif model_size == "float16":
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16) 
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    elif model_size == "ul2":
        model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

        # Initialize accelerator to distribute model across all available GPUs
        accelerator = Accelerator()
        model, tokenizer = accelerator.prepare(model, tokenizer)
    else:
        raise ValueError(f"Invalid model : {model_size}")

    return model, tokenizer

def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate Flan-T5 model on SAT questions')
    parser.add_argument('model_size', choices=['small', 'base', 'large', 'xl', 'xxl', 'eightbitmodel', 'float16', "ul2"], help='Size of T5 model')
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_size)

    # Load data
    prompts = ["""You are a student taking the GRE solving a text completion question.

The question is as follows:
Many find it strange that her writing is thought to be tortuous; her recent essays, although longer than most of her earlier essays, are extremely __________.

You need to pick the word best suited for each blank space:
i) A) painstaking   B) tedious   C) insightful   D) sophisticated   E) clear   

Choose a letter from each of the options given that is correct. Those letters are: """,
"""You are a student taking the GRE solving a text completion question.

The question is as follows:
The unironic representation of objects from everyday life is (i) serious American art of the twentieth century: "high" artists ceded the straightforward depiction of the (ii) __________ to illustrators, advertisers, and packaging designers

You need to pick the word best suited for each blank space:
i) A) missing from   B) valued in   C) crucial to   
ii) D) Beautiful   E) Commonplace   F) Complex   

Choose a letter from each of the options given that is correct. Those letters are: """,
"""You are a student taking the GRE solving a text completion question.

The question is as follows:
There is nothing that (i) _____ scientists more than having an old problem in their field solved by someone from outside. If you doubt this (ii) _____, just think about the (iii) _____ reaction of paleontologists to the hypothesis of Luis Alvarez–a physicist– and Walter Alvarez–a geologist– that the extinction of the dinosaurs was caused by the impact of a large meteor on the surface of the planet

You need to pick the word best suited for each blank space:
i) A) amazes   B) pleases   C) nettles   
ii) D) exposition   E) objurgation   F) observation   
iii) G) contemptuous   H) indifferent   I) insincere   

Choose a letter from each of the options given that is correct. Those letters are: """]

    for i in range(3):
        inputs = tokenizer(prompts[i], return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(inputs, max_new_tokens=10)
        a = tokenizer.decode(outputs[0])
        print(a)
        print()

