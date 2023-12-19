import os
import time
import csv

import torch
from river import datasets
import datetime

from river import naive_bayes
from river import tree, forest, neural_net, neighbors, linear_model
from river import preprocessing
from river import metrics
from river import optim

from river.models import DenseNN
from river.models import get_loss_fn, get_optimizer_fn

model_list = []
model_name = "DenseNN"
model_params = {"hidden_size": 207*12}

model_list.append((model_name, model_params))

print(f"Created {len(model_list)} models")

datasets_list = [
    ["METR_LA",datasets.METR_LA()],
    ["PEMS_BAY",datasets.PEMS_BAY()],
    ["PEMS_03",datasets.PEMS_03()],
    ["PEMS_04",datasets.PEMS_04()]
]

percent = 0

header_row = ['Model', 'Params', 'Dataset', 'Metrics', 'Train Time (s)', 'Mean Learn Time (ms)', 'Mean Predict Time (ms)']

results_path = 'results/results.csv'

file_exists = os.path.isfile(results_path) and os.path.getsize(results_path) > 0

with open(results_path, 'a', newline='') as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(header_row)
    
    total_models = len(model_list) * len(datasets_list)
    current_model = 0
    init_time = time.time()

    for (model_name, params) in model_list:
        
        for dataset_name, dataset in datasets_list:
            model = DenseNN(num_features=dataset.n_features, past_history=dataset.past_history, pred_len=dataset.forecast_horizon, **params)
            # model = DenseNN(input_size=207*12, pred_len=207*6, **params)

            print(model)

            # Constants
            optimizer_name = "sgd"
            lr = 0.01
            loss_fn = "mse"

            # Initialize functions
            loss_fn = get_loss_fn(loss_fn)
            optimizer_fn = get_optimizer_fn(optimizer_name)
            optimizer = optimizer_fn(
                model.parameters(), lr=lr
            )

            scaler = preprocessing.StandardScaler()
            metrics_list = [
                metrics.multioutput.MicroAverage(metrics.MAE()),
                metrics.multioutput.MicroAverage(metrics.RMSE()),
                metrics.multioutput.MicroAverage(metrics.MAPE())
            ]
            learn_time_list = []
            predict_time_list = []
            reading_time_list = []
            finish_time = 0

            start_train_time = time.time()
            for i, (x, y) in enumerate(dataset):
                # Standardize data
                scaler.learn_one(x)
                scaler.learn_one(y)

                x = scaler.transform_one(x)
                y = scaler.transform_one(y)

                # print(f"Training {model_name} on {dataset_name} | {i}/{dataset.n_samples}", end='\r')
                if i > percent * dataset.n_samples:
                # print(f"Training {model_name} on {dataset_name} | {i}/{5000}", end='\r')
                # if i > percent * 5000:
                    if i > 0:
                        reading_time_list.append(time.time() - finish_time)
                    start_pred_time = time.time()

                    # Inference
                    model.eval()

                    x = torch.tensor(list(x.values()))

                    y_pred = model(x)

                    y_pred = y_pred.detach().numpy()
                    y_pred = {k: y_pred[i] for i, k in enumerate(y.keys())}

                    # Update metrics
                    y_unscaled = scaler.inverse_transform_one(y)
                    y_pred_unscaled = scaler.inverse_transform_one(y_pred)

                    for metric in metrics_list:
                        metric.update(y_unscaled, y_pred_unscaled)

                    # Train
                    model.train()
                    optimizer.zero_grad()
                    y_tensor = torch.tensor(list(y.values()))

                    y_pred = model(x)
                    predict_time_list.append(time.time() - start_pred_time)

                    loss = torch.mean((y_pred-y_tensor)**2)

                    # Optimize
                    loss.backward()
                    optimizer.step()
                    
                    start_learn_time = time.time()

                    learn_time_list.append(time.time() - start_learn_time)
                    finish_time = time.time()
                    # print(f"Reading Time: {reading_time_list[-1]:.3f} | Learn Time: {learn_time_list[-1]:.3f} | Predict Time: {predict_time_list[-1]:.3f}", end='\r')
                else:
                    start_learn_time = time.time()

                    x = torch.tensor(list(x.values()))

                    # Train
                    model.train()
                    optimizer.zero_grad()

                    y_tensor = torch.tensor(list(y.values()))

                    y_pred = model(x)

                    loss = torch.mean((y_pred-y_tensor)**2)

                    # Optimize
                    loss.backward()
                    optimizer.step()
                    
                    start_learn_time = time.time()

                    learn_time_list.append(time.time() - start_learn_time)
                    finish_time = time.time()
                    # print(f"Reading Time: {reading_time_list[-1]:.3f} | Learn Time: {learn_time_list[-1]:.3f} | Predict Time: {predict_time_list[-1]:.3f}", end='\r')

                # print the times for each step
                
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

            writer.writerow([model_name, params, dataset_name, metrics_results, train_time, mean_learn_time * 1000, mean_predict_time * 1000])

print()
print("Finished training all models")
