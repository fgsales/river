import time
import csv
from river import datasets
import datetime
import os

from river import naive_bayes
from river import tree, forest, neural_net, neighbors, linear_model
from river import preprocessing
from river import metrics
from river import optim
from river import multioutput, stream

from sklearn import datasets

from sklearn.linear_model import LinearRegression
from river.tree.splitter import TEBSTSplitter

from sklearn.model_selection import ParameterGrid

import numpy as np
import json

def denormalize(data,param):
    return data * (param["max"]-param["min"]) + param["min"]

hoeffding_tree_params_grid = {
    'grace_period': [5, 10, 20, 50],
    'delta': [0.01, 0.1],
    'model_selector_decay': [0.5, 0.7, 0.9],
    'tau': [0.01, 0.1],
}

random_forest_params_grid = {
    'n_models': [3, 5, 10, 20],
    # 'model_selector_decay': [0.5, 0.9],
    # 'grace_period': [5, 20, 50],
    # 'delta': [0.01, 0.1],
    # 'tau': [0.01, 0.1],
}

neighbors_params_grid = {
    'n_neighbors': [3, 5, 10, 20],
    'aggregation_method': ['median', 'weighted_mean'],
}

pa_params_grid = {
    'C': [0.001, 0.01, 0.1, 0.5],
    'mode': [1, 2],
    'eps': [0.001, 0.01, 0.1],
    'learn_intercept': [True],
}


model_list = []

hoeffding_tree_param_combinations = ParameterGrid(hoeffding_tree_params_grid)
random_forest_param_combinations = ParameterGrid(random_forest_params_grid)
neighbors_param_combinations = ParameterGrid(neighbors_params_grid)
pa_param_combinations = ParameterGrid(pa_params_grid)
mlp_hidden_dims = [(5,), (5, 5), (10,), (10, 10), (20,), (20, 20)]

# for params in hoeffding_tree_param_combinations:
#     model = (preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor(**params))
#     model_list.append((model, params))

# for params in hoeffding_tree_param_combinations:
#     model = (preprocessing.StandardScaler() | tree.HoeffdingAdaptiveTreeRegressor(**params))
#     model_list.append((model, params))

# for params in random_forest_param_combinations:
#     model = (preprocessing.StandardScaler() | forest.ARFRegressor(**params))
#     model_list.append((model, params))

for params in mlp_hidden_dims:
    model = "MLPRegressor"
    model_list.append((model, params))

# for params in neighbors_param_combinations:
#     model = (preprocessing.StandardScaler() | neighbors.KNNRegressor(**params))
#     model_list.append((model, params))

# for params in pa_param_combinations:
#     model = (preprocessing.StandardScaler() | linear_model.PARegressor(**params))
#     model_list.append((model, params))

# model_list.append(((preprocessing.StandardScaler() | linear_model.LinearRegression()), {}))

DATASETS = ["ETTh1","ETTh2"]

print(f"Created {len(model_list)} models for {len(DATASETS)} datasets")

datasets_list = []

for DATASET in DATASETS:

    FACTOR = 3

    X_TRAIN_PATH = f"/home/javgal/datasets/{DATASET}/minmax/{FACTOR}/x_train.np.npy"
    Y_TRAIN_PATH = f"/home/javgal/datasets/{DATASET}/minmax/{FACTOR}/y_train.np.npy"
    X_TEST_PATH = f"/home/javgal/datasets/{DATASET}/minmax/{FACTOR}/x_test.np.npy"
    Y_TEST_PATH = f"/home/javgal/datasets/{DATASET}/minmax/{FACTOR}/y_test.np.npy"


    # Read data
    x_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    x_test_full = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    x_train = x_train.reshape(x_train.shape[:2])
    x_test_full = x_test_full.reshape(x_test_full.shape[:2])


    X_full = np.concatenate((x_train, x_test_full), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)

    with open(f'/home/javgal/datasets/{DATASET}/minmax/{FACTOR}/norm_params.json', 'r') as f:
        param = json.load(f)
        for i,serie_ in enumerate(X_full):
            X_full[i] = denormalize(serie_,param[0])
            y_full[i] = denormalize(y_full[i],param[0])

    # dataset_generator = stream.iter_array(
    #     X_full, y_full
    # )

    datasets_list.append((DATASET,X_full, y_full,len(X_full),len(y_full[0])))

percent = 0.2

header_row = ['Model', 'Params', 'Dataset', 'Metrics', 'Train Time (s)', 'Mean Learn Time (ms)', 'Mean Predict Time (ms)']

with open('results/results.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    if os.stat('results/results.csv').st_size == 0:
        writer.writerow(header_row)
    
    total_models = len(model_list) * len(datasets_list)
    current_model = 0
    init_time = time.time()
    rename = False 

    for (model, params) in model_list:
        
        for dataset_name, X_full, y_full, n_samples, n_attributes in datasets_list:
            dataset = stream.iter_array(
                X_full, y_full
            )
            if model == "MLPRegressor":
                rename, value_rename = True, "MLPRegressor"
                activations_list = [neural_net.activations.ReLU for _ in params]
                activations_list.append(neural_net.activations.ReLU)
                activations_list.append(neural_net.activations.Identity)
                model = (preprocessing.StandardScaler() | 
                    neural_net.MLPRegressor(hidden_dims=params, activations=activations_list, optimizer=optim.SGD(1e-3)))

            # model to chain models
            model_train = multioutput.RegressorChain(model, order=list(range(n_attributes)))

            metrics_list = [
                metrics.multioutput.MicroAverage(metrics.MAE()),
                metrics.multioutput.MicroAverage(metrics.RMSE()),
                metrics.multioutput.MicroAverage(metrics.MAPE())
            ]
            learn_time_list = []
            predict_time_list = []

            start_train_time = time.time()
            for i, (x, y) in enumerate(dataset):
                if i > percent * n_samples:
                    start_pred_time = time.time()
                    y_pred = model_train.predict_one(x)
                    predict_time_list.append(time.time() - start_pred_time)

                    for metric in metrics_list:
                        metric.update(y, y_pred)
                    
                    start_learn_time = time.time()
                    model_train.learn_one(x, y)
                    learn_time_list.append(time.time() - start_learn_time)
                else:
                    start_learn_time = time.time()
                    model_train.learn_one(x, y)
                    learn_time_list.append(time.time() - start_learn_time)
                
            current_model += 1
            progress = current_model / total_models
            remaining_models = total_models - current_model
            total_time = time.time() - init_time
            estimated_time = (total_time/current_model) * remaining_models
            print(f"Progress: {progress:.1%} | Estimated Time to Finish: {estimated_time:.2f} seconds", end='\r')

            end_train_time = time.time()
            train_time = end_train_time - start_train_time

            mean_learn_time = sum(learn_time_list) / len(learn_time_list)
            mean_predict_time = sum(predict_time_list) / len(predict_time_list)

            metrics_results = []
            for metric in metrics_list:
                metric_result = metric.get()
                metrics_results.append(f"{metric}")

            writer.writerow([model, params, dataset_name, metrics_results, train_time, mean_learn_time * 1000, mean_predict_time * 1000])

            if rename:
                rename = False
                model = value_rename

print()
print("Finished training all models")
