import copy
import numpy as np
import pickle
import random

import torch
from torch import nn
from torchao.quantization import (
    quantize_,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    int4_weight_only,
)
from torchao.quantization import DEFAULT_INT4_AUTOQUANT_CLASS_LIST
from torchao.quantization.autoquant import AUTOQUANT_CACHE


class DeepNN(nn.Module):
    def __init__(self, d: int, hidden_size: int, depth: int, mode: str = 'special'):
        super().__init__()

        torch.set_default_dtype(torch.float32)

        layers = []
        prev_dim = d
        for _ in range(depth):
            linear = nn.Linear(prev_dim, hidden_size)

            if mode == 'special':
                # Special initialization as in original code
                gain = nn.init.calculate_gain('relu')
                std = gain / np.sqrt(prev_dim)
                nn.init.normal_(linear.weight, mean=0.0, std=std)
                nn.init.zeros_(linear.bias)
            else:
                # Standard PyTorch initialization
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)

            layers.extend([
                linear,
                nn.ReLU()
            ])
            prev_dim = hidden_size

        final_layer = nn.Linear(prev_dim, 1)
        if mode == 'special':
            nn.init.normal_(final_layer.weight, std=0.01)
        else:
            nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()


def evaluate_model(model, X, y, out_idx=None):
    model.eval()
    with torch.no_grad():
        if out_idx is not None:
            y_pred = model(X)[:, out_idx]
        else:
            y_pred = model(X)
        mse = torch.mean((y_pred - y) ** 2).item()
    return mse


def pad_final_linear_layer(model, target_output_features=16):
    """
    Pads the final nn.Linear layer's output features to the target number by adding dummy outputs with zero weights.

    Args:
        model (nn.Module): The trained PyTorch model.
        target_output_features (int): Desired number of output features (must be divisible by 16).

    Returns:
        nn.Module: The modified model with the padded final linear layer.
    """
    # Deep copy the model to avoid in-place modifications
    model_padded = copy.deepcopy(model)

    # Find the last nn.Linear layer
    linear_layers = [module for module in model_padded.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError("No nn.Linear layer found in the model.")

    final_layer = linear_layers[-1]

    current_out_features = final_layer.out_features
    in_features = final_layer.in_features

    if current_out_features >= target_output_features:
        raise ValueError(f"Current output features ({current_out_features}) >= target ({target_output_features}).")

    # Calculate number of dummy outputs to add
    num_to_add = target_output_features - current_out_features

    # Pad the weights with zeros
    with torch.no_grad():
        # Create a tensor of zeros for the new weights
        dummy_weights = torch.zeros((num_to_add, in_features), dtype=final_layer.weight.dtype, device=final_layer.weight.device)
        # Concatenate the existing weights with the dummy weights
        final_layer.weight = nn.Parameter(torch.cat([final_layer.weight, dummy_weights], dim=0))

        if final_layer.bias is not None:
            # Create a tensor of zeros for the new biases
            dummy_bias = torch.zeros(num_to_add, dtype=final_layer.bias.dtype, device=final_layer.bias.device)
            # Concatenate the existing biases with the dummy biases
            final_layer.bias = nn.Parameter(torch.cat([final_layer.bias, dummy_bias], dim=0))

    return model_padded


def quantize_fp16(model):
    model_fp16 = copy.deepcopy(model)
    model_fp16.half()
    return model_fp16


def quantize_int8_weight_only(model):
    model_int8_wo = copy.deepcopy(model)
    quantize_(model_int8_wo, int8_weight_only())
    return model_int8_wo


def quantize_int8_dynamic(model):
    model_int8 = copy.deepcopy(model)
    quantize_(model_int8, int8_dynamic_activation_int8_weight())
    return model_int8


def quantize_int4_weight_only(model):
    padded_model = pad_final_linear_layer(model, target_output_features=16)
    model_int4_wo = copy.deepcopy(padded_model).to(torch.bfloat16)
    quantize_(model_int4_wo, int4_weight_only(group_size=32, use_hqq=False))  # Adjust parameters as needed
    return model_int4_wo


n_train_sizes = [10, 50, 100, 200, 300, 400, 500, 800, 1000, 5000, 10000, 20000]

with open('stair_function/results/no_overlap/test_data.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

results = {
    'fp32': {'train': {}, 'test': {}},
    'fp16': {'train': {}, 'test': {}},
    'int8_wo': {'train': {}, 'test': {}},
    'int8_dyn': {'train': {}, 'test': {}},
    'int4_wo': {'train': {}, 'test': {}},
}

for n_train in n_train_sizes:
    with open(f'stair_function/results/no_overlap/train_data_h800_d4_n{n_train}_lr0.001_standard.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f)

    model = DeepNN(30, 800, 4)
    state_dict = torch.load(f'stair_function/results/no_overlap/final_model_h800_d4_n{n_train}_lr0.001_standard.pt')
    model.load_state_dict(state_dict)

    results['fp32']['train'][n_train] = evaluate_model(model, X_train, y_train)
    results['fp32']['test'][n_train] = evaluate_model(model, X_test, y_test)

    model_fp16 = quantize_fp16(model)
    results['fp16']['train'][n_train] = evaluate_model(model_fp16, X_train.half(), y_train.half())
    results['fp16']['test'][n_train] = evaluate_model(model_fp16, X_test.half(), y_test.half())

    model_int8_wo = quantize_int8_weight_only(model)
    results['int8_wo']['train'][n_train] = evaluate_model(model_int8_wo, X_train, y_train)
    results['int8_wo']['test'][n_train] = evaluate_model(model_int8_wo, X_test, y_test)

    model_int8 = quantize_int8_dynamic(model)
    results['int8_dyn']['train'][n_train] = evaluate_model(model_int8, X_train, y_train)
    results['int8_dyn']['test'][n_train] = evaluate_model(model_int8, X_test, y_test)

    model_int4_wo = quantize_int4_weight_only(model)
    results['int4_wo']['train'][n_train] = evaluate_model(model_int4_wo, X_train.to(torch.bfloat16), y_train.to(torch.bfloat16), out_idx=0)
    results['int4_wo']['test'][n_train] = evaluate_model(model_int4_wo, X_test.to(torch.bfloat16), y_test.to(torch.bfloat16), out_idx=0)

with open('stair_function/results/no_overlap/quantised_results.pkl', 'wb') as f:
    pickle.dump(results, f)
