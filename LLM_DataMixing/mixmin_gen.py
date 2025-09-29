import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

import argparse
import os

def loss_fn(params, logits, lengths):

    final_sum = jax.scipy.special.logsumexp(a = logits.T,axis = 1, b = params)
    final_length_average = jnp.divide(final_sum, lengths)
    loss = -jnp.mean(final_length_average)

    return loss


def update_with_stop(params, logits, lengths, lr):

    grad = jax.grad(loss_fn)(params, logits, lengths)

    denominator = jnp.dot(params, jnp.exp(-lr*grad))

    return jnp.multiply(params, jnp.exp(-lr*grad)) / denominator, grad

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--exp_dir', type = str, default= 'path/to/source_models')
    parser.add_argument('--task', type = str, default='sciq')
    parser.add_argument('--trial', type = int, default = 0)

    args = parser.parse_args()


    exp_dir = args.exp_dir
    #NOTE: expect exp_dir to contain folders for each group containing the logprobs for the source model on the task
    #See Below loop for naming conventions

    task_name = args.task
    trial = args.trial

    save_folder = exp_dir + '/' + f'{task_name}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Collecting Log Probs")
    groups = [0,1,2,3,4,5,6]


    log_probs = []
    lengths = []



    for group in groups:


        output_dir= exp_dir + '/' + f'group_{group}'

        log_probs.append(np.load(output_dir + '/' + f'{task_name}_log_prob.npy'))
        lengths.append(np.load(output_dir + '/' + f'{task_name}_lengths.npy'))
        
    log_probs_np = np.array(log_probs)
    lengths_np = lengths[0]
    gen_losses = -np.mean(np.divide(log_probs_np, lengths_np), axis=1)

    print(gen_losses)

    print("generating and saving train inds")
    train_inds = np.random.choice(2,p = [0.2,0.8], size = len(log_probs_np[0]))

    np.save(save_folder + f'/uncond_inds_{trial}.npy', train_inds)

    train_log_probs = log_probs_np[:,train_inds == 1]
    train_lengths = lengths_np[train_inds ==1]

    test_log_probs = log_probs_np[:,train_inds == 0]
    test_lengths = lengths_np[train_inds == 0]


    print("starting MixMin")
    params = np.ones(len(log_probs_np))/len(log_probs_np)

    lr = 1.0
    max_num_steps = 100
    tol = 1e-4

    loss_list = []
    params_list = [params]

    t = 0
    cont = True

    while t < max_num_steps and cont:
        loss = loss_fn(params, train_log_probs, train_lengths)
        loss_list.append(loss)

        params, grad = update_with_stop(params, train_log_probs, train_lengths, lr)
        params_list.append(params)

        t += 1

        if np.max(np.abs(grad)) < tol:
            cont = False

    gen_test_losses = -np.mean(np.divide(test_log_probs, test_lengths), axis=1)
    print(f"Group Model Losses: {gen_test_losses}")
    test_loss = loss_fn(params_list[-1], test_log_probs, test_lengths)
    print(f"MixMin Ensemble Loss: {test_loss}")

    final_params = params_list[-1]

    
    np.save(save_folder + f'/uncond_weights_{trial}', final_params)


