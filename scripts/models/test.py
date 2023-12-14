import torch
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from river import preprocessing
from torch_geometric_temporal.signal import temporal_signal_split
import pandas as pd
from generic_model import GNN
import pandas as pd
from utils import get_loss_fn, get_optimizer_fn

if __name__=="__main__":

    # Constants
    optimizer_name = "sgd"
    lr = 0.3
    loss_fn = "mse"

    # Load dataset
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

    # Initialize model
    model = GNN(layer_name="DCRNN", n_layers=1, pred_len=1, **{"in_channels": 4, "out_channels": 32, "K": 3})
    
    # Initialize functions
    loss_fn = get_loss_fn(loss_fn)
    optimizer_fn = get_optimizer_fn(optimizer_name)
    optimizer = optimizer_fn(
        model.parameters(), lr=lr
    )
    scaler = preprocessing.StandardScaler()

    loss_avg = 0
    count = 0
    for time, snapshot in enumerate(train_dataset):

        # Standardize data
        input_df = pd.DataFrame(data=torch.concat((snapshot.x, snapshot.y.reshape(-1, 1)), axis=1))
        scaler.learn_many(input_df)
        input_df = scaler.transform_many(input_df)

        input_scaled = torch.tensor(input_df.values)
        x_scaled = input_scaled[:, :snapshot.x.shape[1]]
        y_scaled = input_scaled[:, snapshot.x.shape[1]:]

        # Inference
        model.train()
        optimizer.zero_grad()
        yhat = model(x_scaled, snapshot.edge_index, snapshot.edge_attr)

        # MSE calculation
        loss = torch.mean((yhat-y_scaled)**2)
        
        # Optimize
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
        count += 1

    print("Average MSE train: {:.4f}".format(loss_avg/count))