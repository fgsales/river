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

class NN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, pred_len):
        super(NN, self).__init__()

        self.body = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, pred_len)
        )

    def forward(self, x):
        output = self.body(x)
        return output