import os
import time
import csv
import torch
from river import datasets
from river.models import DenseNN, RNNModel, CNNModel
from river.models import get_loss_fn, get_optimizer_fn
from river import preprocessing
from river import metrics
from river import optim
from multiprocessing import Pool, Manager

from river.models import DenseNN, RNNModel, CNNModel
from river.models import get_loss_fn, get_optimizer_fn

def train_model(model_details):
    model_name, params, dataset_name, dataset, gpu_status_lock, results_path = model_details
    with open(results_path, 'a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header_row)

        if model_name == "DenseNN":
            model = DenseNN(num_features=dataset.n_features, past_history=dataset.past_history, pred_len=dataset.forecast_horizon, **params)
        elif model_name == "RNNModel":
            model = RNNModel(num_features=dataset.n_features, past_history=dataset.past_history, pred_len=dataset.forecast_horizon, **params)
        elif model_name == "CNNModel":
            model = CNNModel(num_features=dataset.n_features, past_history=dataset.past_history, pred_len=dataset.forecast_horizon, **params)
        else:
            raise Exception("Invalid model name")
        
        with gpu_status_lock:
            for gpu_id, status in gpu_status.items():
                if status == 'available':
                    gpu_status[gpu_id] = 'in use'
                    break
            else:
                raise RuntimeError("No available GPU found")

        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        model.to(device)

        update_interval = 1

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

            if update_interval > 0 and i % update_interval == 0:
                print(f"GPU {gpu_id}: Training {model_name} on {dataset_name} | {i}/{dataset.n_samples}")

            if i > percent * dataset.n_samples:
            # print(f"Training {model_name} on {dataset_name} | {i}/{5000}", end='\r')
            # if i > percent * 5000:
                if i > 0:
                    reading_time_list.append(time.time() - finish_time)
                start_pred_time = time.time()

                # Inference
                model.eval()

                x = torch.tensor(list(x.values())).to(device)

                y_pred = model(x)

                y_pred = y_pred.detach().cpu().numpy()
                y_pred = {k: y_pred[i] for i, k in enumerate(y.keys())}

                # Update metrics
                y_unscaled = scaler.inverse_transform_one(y)
                y_pred_unscaled = scaler.inverse_transform_one(y_pred)

                for metric in metrics_list:
                    metric.update(y_unscaled, y_pred_unscaled)

                # Train
                model.train()
                optimizer.zero_grad()
                y_tensor = torch.tensor(list(y.values())).to(device)

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

                x = torch.tensor(list(x.values())).to(device)

                # Train
                model.train()
                optimizer.zero_grad()

                y_tensor = torch.tensor(list(y.values())).to(device)

                y_pred = model(x)

                loss = torch.mean((y_pred-y_tensor)**2)

                # Optimize
                loss.backward()
                optimizer.step()
                
                start_learn_time = time.time()

                learn_time_list.append(time.time() - start_learn_time)
                finish_time = time.time()
                # print(f"Reading Time: {reading_time_list[-1]:.3f} | Learn Time: {learn_time_list[-1]:.3f} | Predict Time: {predict_time_list[-1]:.3f}", end='\r')


        end_train_time = time.time()
        train_time = end_train_time - start_train_time

        mean_learn_time = sum(learn_time_list) / len(learn_time_list)
        mean_predict_time = sum(predict_time_list) / len(predict_time_list)

        metrics_results = []
        for metric in metrics_list:
            metric_result = metric.get()
            metrics_results.append(f"{metric_result}")

        writer.writerow([model_name, params, dataset_name, metrics_results, train_time, mean_learn_time * 1000, mean_predict_time * 1000])

    # Mark the GPU as available again
    with gpu_status_lock:
        gpu_status[gpu_id] = 'available'


    # Once training is complete, return any results you need
    return model_name, params, dataset_name, gpu_id

model_list = []
model_name_list = ["RNNModel", "CNNModel"]
# model_name_list = ["CNNModel"]
model_params = {"hidden_size": 1000}

for model_name in model_name_list:
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

num_gpus = torch.cuda.device_count()
manager = Manager()
log_queue = manager.Queue()
gpu_status = manager.dict({i: 'available' for i in range(num_gpus)})
gpu_status_lock = manager.Lock()

tasks = []
for (model_name, params) in model_list:
    for dataset_name, dataset in datasets_list:
        tasks.append((model_name, params, dataset_name, dataset, gpu_status_lock, results_path))

with Pool(processes=num_gpus) as pool:
    results = pool.map(train_model, tasks)


# Create a pool of workers and execute the tasks
with Pool(processes=num_gpus) as pool:
    results = pool.map(train_model, tasks)

print("\nFinished training all models")