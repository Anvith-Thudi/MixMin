import jax 
import jax.numpy as jnp
import numpy as np
import polaris as po

import datamol as dm

import sklearn
from sklearn.model_selection import train_test_split
import xgboost as xgb
import scipy

import pandas as pd

import os

from sklearn.metrics import average_precision_score
import argparse

import sys


def CE_loss(out,y):

  #we first convert y into one-hot encoding
  y = y.astype(int)
  b = np.zeros((len(y), len(out[0])))
  b[np.arange(len(y)),y] = 1


  return -jnp.log(jnp.sum(jnp.multiply(out,b), axis = 1))


def f_lambda(params, outputs):

    output = 0

    for i in range(len(params)):

        output += outputs[i]*params[i]


    return output

def loss_fn(params, outputs, label):

    loss = jnp.mean(CE_loss(f_lambda(params, outputs), label))

    return loss

def update_with_stop(params, outputs, label, lr):

    grad = jax.grad(loss_fn)(params, outputs, label)

    denominator = jnp.dot(params, jnp.exp(-lr*grad))

    return jnp.multiply(params, jnp.exp(-lr*grad)) / denominator, grad

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Settings')

    parser.add_argument('--n_datasets', type = int, default= 100)
    parser.add_argument('--dataset_ind', type = int, default= 0)
    parser.add_argument('--trial', type = int, default= 1)
    parser.add_argument('--results_folder', type = str, default = 'path/to/saved/results')

    args = parser.parse_args()

    #NOTE: NEED TO RUN PCBA_100k_models.py first

    results_path = args.results_folder
    #NOTE: see below for naming convention for things in results folder


    print(f"On Dataset {args.dataset_ind}")


    print("Loading Dataset")
    test_features = np.load(results_path + '/' + f'test_features_{args.dataset_ind}.npy')
    test_labels = np.load(results_path + '/' + f'test_labels_{args.dataset_ind}.npy')

    train_features = np.load(results_path + '/' + f'train_features_{args.dataset_ind}.npy')
    train_labels = np.load(results_path + '/' + f'train_labels_{args.dataset_ind}.npy')

    val_features = np.load(results_path + '/' + f'val_features_{args.dataset_ind}.npy')
    val_labels = np.load(results_path + '/' + f'val_labels_{args.dataset_ind}.npy')


    #Loading models
    print("Loading Models")
    assay_inds = np.load(results_path + '/' + 'assay_inds.npy')

    baseline_models = []
    ind_models = []

    for i in range(len(assay_inds)):
        ind_model_path = f"ind_model_{i}.model"
        bst = xgb.XGBClassifier()
        bst.load_model(results_path + '/' + ind_model_path)
        ind_models.append(bst)



        baseline_model_path = f"baseline_model_{i}.model"
        bst = xgb.XGBClassifier()
        bst.load_model(results_path + '/' + baseline_model_path)
        baseline_models.append(bst)

    
    #Now doing mixmin

    mixmin_weights = []
    mixmin_losses = []

    lr = 1.0
    max_num_steps = 100
    tol = 1e-4


    print("defining mixmin alg inputs")
    optim_funcs = ind_models[:args.dataset_ind] + ind_models[args.dataset_ind+1:]
    x = train_features
    y = train_labels

    outputs = []

    for f in optim_funcs:
        outputs.append(f.predict_proba(x))

    params = jnp.ones(len(optim_funcs))*(1/len(optim_funcs))

    loss_list = []
    params_list = [params]

    t = 0
    cont = True

    print("running mixmin")

    while t < max_num_steps and cont:
        loss = loss_fn(params, outputs,y)
        loss_list.append(loss)

        params, grad = update_with_stop(params, outputs, y, lr)
        params_list.append(params)

        t += 1

        if np.max(np.abs(grad)) < tol:
            cont = False

    print("saving mixmin stats")
    mixmin_weights = params_list[-10:-1]
    mixmin_losses = loss_list[-10:-1]

    np.save(results_path + '/' + f'mixmin_weights_{args.dataset_ind}', np.array(mixmin_weights))




    print("Now Getting Results")
    
    baseline_model = baseline_models[args.dataset_ind]

    baseline_probs = baseline_model.predict_proba(test_features)[:,1]
    baseline_AP_score = average_precision_score(test_labels, baseline_probs)

    #getting the mixmin and prodmin outputs
    optim_outputs = []
    
    for f in optim_funcs:
        probs = f.predict_proba(test_features)
        optim_outputs.append(np.copy(probs))



    #MIXMIN Ensemble
    mixmin_outputs = f_lambda(mixmin_weights[-1], optim_outputs)
    mixmin_probs = mixmin_outputs[:,1]
    mixmin_AP_score = average_precision_score(test_labels, mixmin_probs)


    np.save(results_path + "/" + f"mixmin_results_{args.dataset_ind}", np.array([baseline_AP_score, mixmin_AP_score]))










