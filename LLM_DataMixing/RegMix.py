import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import lightgbm as lgb

import argparse
import os

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--trial', type = int, default= 0)
    parser.add_argument('--n_models', type = int, default= 7)
    parser.add_argument('--task', type = str, default='piqa')
    parser.add_argument('--exp_dir', type = str, default = 'path/to/random_models_folder')
    args = parser.parse_args()

    trial = args.trial
    task_name = args.task
    lr = args.lr
    n_models = args.n_models

    exp_dir = args.exp_dir
    #NOTE: assume log probs, mixture for random mixture models are saved in exp_dir under train_{t}
    #See Below loop for naming conventions

    save_folder = exp_dir + '/' + f'{task_name}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Collecting Log Probs")
    r_trials = list(np.arange(start = 0, stop = n_models))


    log_probs = []
    lengths = []
    mixtures = []

    for t in r_trials:


        output_dir= exp_dir + '/' + f'train_{t}'

        log_probs.append(np.load(output_dir + '/' + f'{task_name}_log_prob.npy'))
        lengths.append(np.load(output_dir + '/' + f'{task_name}_lengths.npy'))

        mixtures.append(np.load(exp_dir + '/'+ f'mixture_{t}.npy'))
            
    log_probs_np = np.array(log_probs)
    lengths_np = lengths[0]
    gen_losses = -np.mean(np.divide(log_probs_np, lengths_np), axis=1)

    print(gen_losses)

    print("generating and saving train inds")
    train_inds = np.random.choice(2,p = [0.2,0.8], size = len(log_probs_np[0]))

    np.save(save_folder + f'/RegMix_inds_{n_models}_{trial}.npy', train_inds)

    train_log_probs = log_probs_np[:,train_inds == 1]
    train_lengths = lengths_np[train_inds ==1]

    test_log_probs = log_probs_np[:,train_inds == 0]
    test_lengths = lengths_np[train_inds == 0]

    print("Running RegMix")
    

    gen_train_losses = -np.mean(np.divide(train_log_probs, train_lengths), axis=1)
    print(f"Random Model Losses: {gen_train_losses}")
    
    #Added min_data parameters to allow training over smaller datasets (otherwise was constant function)
    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1','l2'],
        "num_iterations": 10, 
        'seed': 42,
        'learning_rate': 1e-2,
        "verbosity": -1,
        "min_data_in_leaf": 1,
        "min_data_in_bin": 1
    }


    cutoff = int((2/3)*n_models)
    X_train = np.array(mixtures)[:cutoff]
    target_train = gen_train_losses[:cutoff]

    X_test = np.array(mixtures)[cutoff:]
    target_test = gen_train_losses[cutoff:]

    np.random.seed(42)

        
    gbm = lgb.LGBMRegressor(**hyper_params)

    reg = gbm.fit(X_train, target_train
                , eval_set=[(X_test, target_test)],
            eval_metric='l2', callbacks=[
            lgb.early_stopping(stopping_rounds=3, verbose=False),
        ])
    
    np.random.seed(42)
    
    #Changed below to natural SlimPajama distribution
    prior_dist = np.array([52.2, 26.7, 5.2, 4.2, 4.6,3.8,3.3]) /100.0


    samples = np.random.dirichlet(prior_dist * 1, 100000)

    simulation = reg.predict(samples)
    k = 128
    top_k_samples = samples[np.argsort(simulation)[0:k]]

    # you can get the optimal data mixture by taking the average of top-k samples
    optimal_data_mixture = np.mean(top_k_samples, axis=0)
    print(optimal_data_mixture)
    

    
    np.save(save_folder + f'/RegMix_weights_{n_models}_{trial}', optimal_data_mixture)


