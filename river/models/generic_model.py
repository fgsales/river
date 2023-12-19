import torch
# import torch_geometric_temporal
import inspect
import torch.nn.functional as F

# recurrent_layers = inspect.getmembers(torch_geometric_temporal.nn.recurrent, inspect.isclass)
# attn_layers = inspect.getmembers(torch_geometric_temporal.nn.attention, inspect.isclass)
# LAYER_MODULES = {name: class_obj for name, class_obj in [*recurrent_layers, *attn_layers]}


def calculate_output_shape(**kwargs):

    out_shape = 1
    if "out_channels" in kwargs:
        out_shape *= kwargs["out_channels"]
    return out_shape


class GNN(torch.nn.Module):
    def __init__(self, layer_name, n_layers, pred_len, **kwargs):
        super(GNN, self).__init__()

        base_layer = LAYER_MODULES[layer_name]

        layers = []
        for i in range(n_layers):
            if i>0:
                kwargs["in_channels"] = kwargs["out_channels"]
                
            layers.append(base_layer(**kwargs))

        self.body = torch.nn.ModuleList(layers)
        self.output = torch.nn.Linear(calculate_output_shape(**kwargs), pred_len)
        

    def forward(self, x, edge_index, edge_weight):

        for layers in self.body:
            x = layers(x, edge_index, edge_weight)
            x = F.relu(x)

        output = self.output(x)

        return output

class DenseNN(torch.nn.Module):
    def __init__(self, hidden_size, num_features, past_history=12, pred_len=6):
        super(DenseNN, self).__init__()
        input_size = num_features * past_history
        output_size = num_features * pred_len

        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc_1(x)
        out = self.relu(out)
        output = self.fc_2(out)
        return output
    
class RNNModel(torch.nn.Module):
    def __init__(self, hidden_size, num_features, past_history=12, pred_len=6):
        super(RNNModel, self).__init__()
        output_size = num_features * pred_len

        self.rnn = torch.nn.RNN(num_features, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * past_history, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    
class CNNModel(torch.nn.Module):
    def __init__(self, hidden_size, num_features, past_history=12, pred_len=6):
        super(CNNModel, self).__init__()
        output_size = num_features * pred_len

        self.layer1 = torch.nn.Conv1d(num_features, hidden_size, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(hidden_size, output_size // past_history, kernel_size=3, stride=1, padding=1)
        self.fc = torch.nn.Linear(past_history, pred_len)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1, past_history)
        out = self.fc(out)
        return out.view(out.size(0), -1)