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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Settings')

    parser.add_argument('--n_datasets', type = int, default= 10)
    parser.add_argument('--test_split', type = float, default= 0.2)
    parser.add_argument('--val_split', type = float, default= 0.2)
    parser.add_argument('--trial', type = int, default= 1)
    parser.add_argument('--results_folder', type = str, default = 'path/to/saved/results')
    parser.add_argument('--data_folder', type = str, default = 'path/to/dataset')


    args = parser.parse_args()

    print("making directory")

    results_path = args.results_folder
    if not os.path.exists(results_path):
        
        print("making directory")
        os.makedirs(results_path)

    #load all the data

    print("loading data")
    data_folder = args.data_folder

    #NOTE: Expect below in the data folder
    inputs_all = np.load(data_folder + 'pcba_100k_input_fp.npy', allow_pickle=True)
    labels_all = np.load(data_folder + 'pcba_100k_label.npy', allow_pickle = True)

    labels_all = labels_all.astype(float)


    #Define datasets based on n_datasets
    print("defining datasets")
    datasets = []
    assay_inds = []
    num_positives_list = []

    num_negatives_list = []

    for i in range(args.n_datasets):
        label_data = labels_all[:, i]
        cleaned_labels = label_data[~np.isnan(label_data)]
        cleaned_inputs = inputs_all[~np.isnan(label_data)]

        num_positives = np.sum(cleaned_labels == 1.0)
        num_negatives = np.sum(cleaned_labels == 0.0)

        if num_positives == 0 or num_negatives == 0:
            print(f"{i} has no positive or no negative instances")

        else:
            assay_inds.append(i)
            datasets.append([cleaned_inputs, cleaned_labels])
            num_positives_list.append(num_positives)

            num_negatives_list.append(np.sum(cleaned_labels == 0.0))

    print("saving assay inds")

    np.save(results_path + '/' + 'assay_inds', np.array(assay_inds))
    np.save(results_path + '/' + 'num_positives', np.array(num_positives_list))


    #saving models for each setting
    n_estimators_list = [10,50,100]
    max_depth_list = [4,6,8]

    best_settings = []

    for i in range(len(datasets)):
        print(i)

        if os.path.exists(results_path + '/' + f"ind_model_{i}.model"):
            print("already done")

        else:
            features = datasets[i][0]
            labels = datasets[i][1]


            train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size= args.test_split)
            new_train_features, val_features, new_train_labels, val_labels = train_test_split(train_features, train_labels, test_size= args.val_split)


            new_train_dmatrix = xgb.DMatrix(data=new_train_features,label=new_train_labels)

            #Now doing cross-val on new_train
            perf_list = []
            setting_list = []

            for n_estimators in n_estimators_list:
                for max_depth in max_depth_list:

                    param = {"max_depth": max_depth, "learning_rate": 0.1, "objective":"binary:logistic"}
                    cv = xgb.cv(
                        param,
                        new_train_dmatrix,
                        num_boost_round= n_estimators,
                        nfold=5,
                        metrics={"aucpr"},
                    )

                    perf_list.append(np.mean(cv.head()['test-aucpr-mean']))
                    setting_list.append([n_estimators, max_depth])

            best_setting = np.argmax(np.array(perf_list))
            best_n_estimators = setting_list[best_setting][0]
            best_max_depth = setting_list[best_setting][1]

            best_settings.append([best_n_estimators, best_max_depth])
            np.save(results_path + '/' + f'best_setting_{i}', np.array([best_n_estimators, best_max_depth]))

            #saving train, test, and val sets

            np.save(results_path + '/' + f'test_features_{i}', test_features)
            np.save(results_path + '/' + f'test_labels_{i}', test_labels)

            np.save(results_path + '/' + f'train_features_{i}', new_train_features)
            np.save(results_path + '/' + f'train_labels_{i}', new_train_labels)

            np.save(results_path + '/' + f'val_features_{i}', val_features)
            np.save(results_path + '/' + f'val_labels_{i}', val_labels)



            
            bst = xgb.XGBClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, learning_rate=0.1, objective='binary:logistic')
            bst.fit(train_features, train_labels)

            baseline_model_path = f"baseline_model_{i}.model"
            bst.save_model(results_path + '/' + baseline_model_path)


            #training and saving individual model for MixMin

            ind_bst = xgb.XGBClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, learning_rate=0.1, objective='binary:logistic')
            ind_bst.fit(new_train_features, new_train_labels)

            ind_model_path = f"ind_model_{i}.model"
            ind_bst.save_model(results_path + '/' + ind_model_path)



