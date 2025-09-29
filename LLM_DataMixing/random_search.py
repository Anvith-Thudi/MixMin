import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--trial', type = int, default= 0)
    parser.add_argument('--task', type = str, default='sciq')
    parser.add_argument('--exp_dir', type = str, default = 'path/to/random_models_folder')
    parser.add_argument('--lr', type = float, default= 5e-3)
    args = parser.parse_args()

    trial = args.trial
    task_name = args.task
    lr = args.lr

    exp_dir = args.exp_dir
    #NOTE: assume log probs, mixture for random mixture models are saved in exp_dir under train_{t}
    #See Below loop for naming conventions

    save_folder = exp_dir + '/' + f'{task_name}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Collecting Log Probs")
    r_trials = list(np.arange(start = trial*7, stop = (trial+1)*7))


    log_probs = []
    lengths = []

    for t in r_trials:


        output_dir= exp_dir + '/' + f'train_{t}'

        log_probs.append(np.load(output_dir + '/' + f'{task_name}_log_prob.npy'))
        lengths.append(np.load(output_dir + '/' + f'{task_name}_lengths.npy'))
        
    log_probs_np = np.array(log_probs)
    lengths_np = lengths[0]
    gen_losses = -np.mean(np.divide(log_probs_np, lengths_np), axis=1)

    print(gen_losses)

    print("genering and saving train inds")
    train_inds = np.random.choice(2,p = [0.2,0.8], size = len(log_probs_np[0]))

    np.save(save_folder + f'/inds_{trial}.npy', train_inds)

    train_log_probs = log_probs_np[:,train_inds == 1]
    train_lengths = lengths_np[train_inds ==1]

    test_log_probs = log_probs_np[:,train_inds == 0]
    test_lengths = lengths_np[train_inds == 0]


    print("Picking Best Performance")
    

    gen_train_losses = -np.mean(np.divide(train_log_probs, train_lengths), axis=1)
    print(f"Random Model Losses: {gen_train_losses}")
    

    best_ind = np.argmin(gen_train_losses)
    best_t = r_trials[best_ind]

    best_random_mixture = np.load(exp_dir + '/' + f'mixture_{best_t}.npy')

    
    np.save(save_folder + f'/random_weights_{trial}', best_random_mixture)


