from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer 
import numpy as np
import evaluate

import torch
from evaluate import evaluator

from transformers import AutoModelForCausalLM, AutoConfig, DefaultDataCollator
from datasets import Dataset, IterableDataset
import os

from tqdm import tqdm
import concurrent.futures

import argparse

os.environ['TOKENIZERS_PARALLELISM']="false"
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_distributed =  world_size > 1
is_main_process = not is_distributed or int(os.environ.get("RANK", 0)) == 0
print("Distributed", is_distributed)

#torch.distributed.init_process_group()


def sample_from_vec(prob_vector: np.array, batch_size: int, ctx_len: int, memmapped_array: np.array, start_map: np.array, len_map: np.array, gen: np.random.Generator = np.random.Generator(np.random.PCG64())):
    # samples tokens in a weighted way from documents.
    # samples a doc proportionally to prob_vector.
    # within each doc, sample a window of ctx_len uniformly at random.
    # returns the sampled batch of token indices
    #assert(np.min(len_map) >= ctx_len)  # can kill this if slow..
    # get the document ids
    #doc_ids = np.array(random.choices(range(len(prob_vector)), weights=prob_vector, k=batch_size)) #random.choices is slightly faster than numpy
    doc_ids = gen.choice(len(prob_vector), p=prob_vector, size=batch_size)
    # now get the offsets -
    offset_ids = np.random.randint(len_map[doc_ids] - ctx_len + 1)
    start_points = start_map[doc_ids] + offset_ids
    # do some fancy reshaping to do vectorized indexing
    flattened_idx = np.add.outer(start_points, np.arange(ctx_len)).reshape(ctx_len*batch_size)
    sampled_batch = memmapped_array[flattened_idx].reshape(batch_size, ctx_len)
    return torch.LongTensor(sampled_batch), torch.ones(sampled_batch.shape)

def get_dataset(prob_vector:np.array, ctx_len: int, memmaped_file: str, start_map: np.array, len_map: np.array, max_tokens: int, batch_size = 10000):
    def gen():
        rng = np.random.Generator(np.random.PCG64())
        while True:
            temp_memmap = np.memmap(memmaped_file, dtype='int32', mode='r', shape=(max_tokens))  # reinitialize memmap for memory
            sampled_batches, masks = sample_from_vec(prob_vector, batch_size, ctx_len, temp_memmap, start_map, len_map, rng)
            for i in range(batch_size):
                yield {
                    "input_ids": sampled_batches[i,:].squeeze(),
                    "labels": sampled_batches[i,:].squeeze(),
                    "attention_mask": masks[i,:].squeeze()
                }
    print('get_dataset')
    return IterableDataset.from_generator(gen)

import time
def get_dataset_async(prob_vector: np.array, ctx_len: int, memmaped_file: str, start_map: np.array, len_map: np.array,
                      max_tokens: int, batch_size = 10000):
    # async version of the above - used to overlap reads and GPU computation
    def gen():
        rng = np.random.Generator(np.random.PCG64())
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_batch = executor.submit(sample_from_vec, prob_vector, batch_size, ctx_len,
                                           np.memmap(memmaped_file, dtype='int32', mode='r', shape=(max_tokens)),
                                           start_map, len_map, rng)

            while True:
                start = time.time()
                # Wait for the future to complete and get the result
                sampled_batches, masks = future_batch.result()

                # Submit the next batch generation
                future_batch = executor.submit(sample_from_vec, prob_vector, batch_size, ctx_len,
                                               np.memmap(memmaped_file, dtype='int32', mode='r', shape=(max_tokens)),
                                               start_map, len_map, rng)

                end = time.time()
                print('batch overhead '+str(end-start)+'(s)')
                for i in range(batch_size):
                    yield {
                        "input_ids": sampled_batches[i,:].squeeze(),
                        "labels": sampled_batches[i,:].squeeze(),
                        "attention_mask": masks[i,:].squeeze()
                    }

    print('get_dataset')
    return IterableDataset.from_generator(gen)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--tokens', type = int, default= 3200000000)
    parser.add_argument('--mixture', type = str, default= 'path/to/mixure_weights')
    parser.add_argument('--output_dir', type = str, default= 'path/to/model_checkpoint' )
    parser.add_argument('--data_dir', type = str, default= 'path/to/dataset' )
    parser.add_argument('--lr', type = float, default= 5e-3)
    parser.add_argument('--model_size', type = int, default= 160)

    args = parser.parse_args()

    lr = args.lr
    print(f"learning rate {lr}")


    master_port = 29513

    print("initializing dataset")

    data_dir=args.data_dir
    file_prefix = data_dir + f'/balanced_{3200000000}'
    start_idx = np.load(file_prefix + "_start.npy")
    len_idx = np.load(file_prefix + "_len.npy")
    max_tokens = np.load(file_prefix + "_metadata.npy")
    domain_ids = np.load(file_prefix + "_id.npy")

    print("Defining Sampling Vector")

    mix_path = args.mixture
    mix_w = np.load(mix_path)

    print(f"Mixture Weights: {mix_w}")
    prob_vector = np.zeros(len(domain_ids))
    for g in range(7):
        g_inds = domain_ids == g
        n_g = np.sum(g_inds)
        prob_vector[g_inds] = mix_w[g] / n_g

    #Normalize to account for precision errors

    prob_vector = prob_vector / np.sum(prob_vector)


    train_dataset = get_dataset_async(prob_vector=prob_vector, ctx_len=1024, memmaped_file=file_prefix + ".mmap", start_map=start_idx, len_map=len_idx, max_tokens=max_tokens*7, batch_size=1000)



    print("initializing trainer")
    # a toy training run 

    if args.model_size == 410:
        print("Using 410M")
        output_dir= args.output_dir
        model_id = 'EleutherAI/pythia-410m'
        batch_size = 32

    else:
        print("Using 160M")
        output_dir= args.output_dir
        model_id = 'EleutherAI/pythia-160m'
        batch_size = 64

    tokens_per_gpu = args.tokens/world_size
    steps = int(tokens_per_gpu / (1024*batch_size))

    pythia_config = AutoConfig.from_pretrained(model_id)
    pythia_m = AutoModelForCausalLM.from_config(pythia_config)

    #Defining Hyperparameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        local_rank=0,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="epoch",
        push_to_hub = False,
        learning_rate = lr, 
        per_device_train_batch_size= batch_size,
        max_steps=steps,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        #torch_compile=True,
        #torch_compile_mode="max-autotune",
        adam_epsilon=1e-8,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        ddp_find_unused_parameters=False,
        ddp_backend="nccl"
    )

    data_collator = DefaultDataCollator(return_tensors="pt")

    trainer = Trainer(
        model=pythia_m,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("starting training")

    trainer.train()
