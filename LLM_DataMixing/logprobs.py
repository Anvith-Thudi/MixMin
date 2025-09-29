
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

import pickle

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

    mixture = args.mixture
    tokens = args.tokens
    task_name = args.task
    world_size = 3


    #NOTE: expect output dir to contain a result folder generated from lm_eval.py for non openwebmath tasks

    print(f'Doing Task {task_name}')

    if args.model_size == 410:
        print("Using 410M")
        tokens = 8200000000
        output_dir= args.output_dir
        model_id = 'EleutherAI/pythia-410m'
        batch_size = 32
        print(f"Tokens {tokens}")

    else:
        print("Using 160M")
        tokens = 3200000000
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


    if task_name != 'openmath_10000':

        def tokenize_function(examples):
            return tokenizer(examples, padding="do_not_pad", truncation=False)
        
        results = pd.read_pickle(output_dir + '/' + f'{task_name}_results.pkl')

        log_probs = []
        lengths = []

        print("geting total logprobs")


        for i in range(len(results)):
            print(f"On input {i} out of {len(results)}")

            y = results['target'][i]
            input_str = ''.join(list(results['arguments'][i][y]))
            input = tokenize_function(input_str)
            input = torch.tensor([input['input_ids']]).to('cuda')

            out = model(input, labels = input)
            logs = out.logits[0]
            soft_probs = torch.softmax(logs, axis=1)

            sel_probs = soft_probs[torch.arange(len(input[0])-1),input[0,1:]].detach().cpu()
            sel_probs = sel_probs.type(torch.float).numpy()

            loss = out.loss.detach().cpu()

            length = len(sel_probs)
            log_probs.append(-loss.item()*length)
            lengths.append(length)

        log_probs_np = np.array(log_probs)
        lengths_np = np.array(lengths)

        np.save(output_dir + '/' + f'{task_name}_log_prob.npy', log_probs_np)
        np.save(output_dir + '/' + f'{task_name}_lengths.npy', lengths_np)

        print("geting input logprobs")

        log_probs_input = []
        lengths_input = []

        for i in range(len(results)):
            print(f"On input {i} out of {len(results)}")

            y = results['target'][i]
            #just taking the question string
            input_str = results['arguments'][i][y][0]
            input = tokenize_function(input_str)
            input = torch.tensor([input['input_ids']]).to('cuda')


            out = model(input, labels = input)
            logs = out.logits[0]
            soft_probs = torch.softmax(logs, axis=1)
            sel_probs = soft_probs[torch.arange(len(input[0])-1),input[0,1:]].detach().cpu()
            sel_probs = sel_probs.type(torch.float).numpy()

            loss = out.loss.detach().cpu()

            length = len(sel_probs)
            log_probs_input.append(-loss.item()*length)
            lengths_input.append(length)

        log_probs_input_np = np.array(log_probs_input)
        lengths_input_np = np.array(lengths_input)

        np.save(output_dir + '/' + f'{task_name}_input_log_prob.npy', log_probs_input_np)
        np.save(output_dir + '/' + f'{task_name}_input_lengths.npy', lengths_input_np)

    else:

        filename = 'path/to/openwebmath_tokens_first_10000_documents'
        with open(f'{filename}.pkl', 'rb') as handle:
            loaded_tokens = pickle.load(handle)

        log_probs = []
        lengths = []

        print("geting total logprobs")


        for i,tokens in enumerate(loaded_tokens):
            print(f"On input {i} out of {len(loaded_tokens)}")

            print(len(tokens))

            #For bigger models may run out of memory for long documents
            if len(tokens) < 40000 or args.model_size == 160:
        
                input = torch.unsqueeze(tokens, dim=0).to('cuda')

                out = model(input, labels = input)
                logs = out.logits[0]
                soft_probs = torch.softmax(logs, axis=1)
                sel_probs = soft_probs[torch.arange(len(input[0])-1),input[0,1:]].detach().cpu()
                sel_probs = sel_probs.type(torch.float).numpy()

                loss = out.loss.detach().cpu()

                length = len(sel_probs)
                log_probs.append(-loss.item()*length)
                lengths.append(length)

            else:
                #FOUND this condition only happens ~10 times and did not affect results
                print("repeating prev: too long")
                #just repeat previous logprobs for now. Need to have same length as 160M 
                log_probs.append(-loss.item()*length)
                lengths.append(length)

        

        log_probs_np = np.array(log_probs)
        lengths_np = np.array(lengths)

        np.save(output_dir + '/' + f'{task_name}_log_prob.npy', log_probs_np)
        np.save(output_dir + '/' + f'{task_name}_lengths.npy', lengths_np)