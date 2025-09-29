
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer 
import numpy as np
import evaluate

import lm_eval
from lm_eval.models.huggingface import HFLM

import torch
from evaluate import evaluator

from transformers import AutoModelForCausalLM, AutoConfig, DefaultDataCollator
from datasets import Dataset

import os
import pandas as pd
import argparse

class HFLM_Local(HFLM):
    def get_model_info(self):
        return {}
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--tokens', type = int, default= 3200000000)
    parser.add_argument('--output_dir', type = str, default= 'path/to/model')
    parser.add_argument('--task', type = str, default='sciq')
    parser.add_argument('--model_size', type = int, default= 160)

    args = parser.parse_args()


    tokens = args.tokens
    task_name = args.task
    lr = args.lr
    world_size = 3

    if args.model_size == 410:
        print("Using 410M")
        output_dir= args.output_dir
        model_id = 'EleutherAI/pythia-410m'
        batch_size = 32
        print(f"Tokens {tokens}")

    else:
        print("Using 160M")
        output_dir=args.output_dir
        model_id = 'EleutherAI/pythia-160m'
        batch_size = 64
        print(f"Tokens {tokens}")


    tokens_per_gpu = tokens/world_size
    steps = int(tokens_per_gpu / (1024*batch_size))

    model= AutoModelForCausalLM.from_pretrained(output_dir +'/'+f'checkpoint-{steps}', 
                                                torch_dtype=torch.bfloat16,
                                                device_map="cuda")
    

    model_id = 'EleutherAI/pythia-160m'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    eval_model = HFLM_Local(pretrained=model, tokenizer = tokenizer)

    results = lm_eval.simple_evaluate(
            model=eval_model,
            tasks=[task_name],
            batch_size="auto",
            limit=5000,
            bootstrap_iters=1000,
            log_samples=True
        )
    
    results_df = pd.DataFrame(results['samples'][task_name])

    results_df.to_pickle(output_dir + '/' + f'{task_name}_results.pkl')
    
