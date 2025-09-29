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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Settings')

    parser.add_argument('--dataset_ind', type = int, default= 0)
    parser.add_argument('--trial', type = int, default= 1)
    parser.add_argument('--results_folder', type = str, default = 'path/to/saved/results')

    args = parser.parse_args()

    results_path = args.results_folder
    #NOTE: see below for naming convention for things in results folder (pcba_100k_models.py creates the files)


    assay_inds = np.load(results_path + '/' + 'assay_inds.npy')

    n_assays = len(assay_inds)

    #Getting the Dataset
    print("getting the data")

    train_features = []
    train_labels = []


    dataset_ind = args.dataset_ind

    for i in range(n_assays):
        if i != dataset_ind:


            train_features.append(np.load(results_path + '/' + f'train_features_{i}.npy'))
            train_labels.append(np.load(results_path + '/' + f'train_labels_{i}.npy'))


    train_features_np = np.concatenate(train_features)
    train_labels_np = np.concatenate(train_labels)


    #Now training the model
    print("Now Training the model")

    n_estimators = 100
    max_depth =6
    lr =0.1

    bst = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, objective='binary:logistic')
    bst.fit(train_features_np, train_labels_np)

    model_path = f"natural_data.model"
    bst.save_model(results_path + '/' + model_path)

    #Now collecting individual performances

    print("getting performance")

    test_features = np.load(results_path + '/' + f'test_features_{dataset_ind}.npy')
    test_labels = np.load(results_path + '/' + f'test_labels_{dataset_ind}.npy')

    probs = bst.predict_proba(test_features)[:,1]
    AP_score = average_precision_score(test_labels, probs)

    np.save(results_path + "/" + f"natural_results_{args.dataset_ind}", np.array([AP_score]))

