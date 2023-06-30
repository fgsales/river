import time
import csv
from river import datasets
import datetime

# Import models and metrics from river
from river import naive_bayes
from river import tree
from river import preprocessing
from river import metrics

from sklearn.linear_model import LinearRegression
from river.tree.splitter import TEBSTSplitter

from sklearn.model_selection import ParameterGrid

# Define your models, datasets and metrics here
hoeffding_tree_params_grid = {
    'grace_period': [5, 20, 50, 100, 200],
    'max_depth': [None, 1, 2, 3, 5, 10],
    'delta': [0.001, 0.01, 0.1],
    'tau': [0.01, 0.1, 1.0],
    'leaf_prediction': ['mean', 'model', 'adaptive'],
    'model_selector_decay': [0.1, 0.5, 0.9],
    'merit_preprune': [True, False]
}

model_list = []

# Generate all parameter combinations
hoeffding_tree_param_combinations = ParameterGrid(hoeffding_tree_params_grid)

# Create models with different parameter combinations
for params in hoeffding_tree_param_combinations:
    model = (preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor(**params))
    model_list.append((model, params))

    # print(f"Created model with params: {params}")

print(f"Created {len(model_list)} models")

datasets_list = [
    ["Trump",datasets.TrumpApproval()],
    # ["Airline",datasets.AirlinePassengers()],
    ["Banana",datasets.Bananas()],
    # ["Bikes",datasets.Bikes()],
    ["ChickWeights",datasets.ChickWeights()],
    # ["Restaurants",datasets.Restaurants()],
    # ["Taxis",datasets.Taxis()],
    # ["WaterFlow",datasets.WaterFlow()],
]

metrics_list = [
    metrics.MAE(),
    metrics.MAPE(),
    metrics.RMSE()
]

percent = 0.2

# Define the header row
header_row = ['Model', 'Params', 'Dataset', 'Metrics', 'Train Time']

# Open the CSV file for writing
with open('results/results.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(header_row)
    writer = csv.writer(file)
    
    # Iterate over models
    for (model,params) in model_list:
        
        # Iterate over datasets
        for dataset_name,dataset in datasets_list:
            # Training and testing time measurement
            start_train_time = time.time()
            for i, (x, y) in enumerate(dataset):
                if i > percent * dataset.n_samples:
                    y_pred = model.predict_one(x)
                    for metric in metrics_list:
                        metric.update(y, y_pred)
                    model.learn_one(x, y)
                else:
                    model.learn_one(x, y)
            end_train_time = time.time()
            train_time = end_train_time - start_train_time
            
            
            # Get metric result
            metrics_results = []
            for metric in metrics_list:
                metric_result = metric.get()
                metrics_results.append(f"{metric}")

            # Write results to CSV
            writer.writerow([model, params, dataset_name, metrics_results, train_time])