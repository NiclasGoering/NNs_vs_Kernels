#!/usr/bin/env python3
import numpy as np
import torch
from typing import List, Set
import random
from functools import partial

# Define print globally first
print = partial(print, flush=True)

class MSPFunction:
    def __init__(self, P: int, sets: List[Set[int]]):
        self.P = P
        self.sets = sets

        # Verify MSP property
        # for i in range(1, len(sets)):
        #     prev_union = set().union(*sets[:i])
        #     diff = sets[i] - prev_union
        #     if len(diff) > 1:
        #         raise ValueError(f"Not an MSP: Set {sets[i]} adds {len(diff)} new elements: {diff}")

    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        result = torch.zeros(batch_size, device=z.device)

        for S in self.sets:
            term = torch.ones(batch_size, device=z.device)
            for idx in S:
                term = term * z[:, idx]
            result = result + term

        return result

def main():

    # Parameters
    P = 8
    d = 30
    n_test = 1000
    n_train_sizes = [10, 50, 100, 200, 300, 400, 500, 800, 1000, 5000, 10000, 20000]


    # Define the specific MSP sets
    msp_sets = [{7}, {2, 7}, {0, 2, 7}, {4, 5, 7}, {1}, {0, 4}, {3, 7}, {0, 1, 2, 3, 4, 6, 7}]


    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(44)

    # Initialize MSP function
    msp = MSPFunction(P, msp_sets)

    # Generate test data (fixed)
    X_test = 2 * torch.bernoulli(0.5 * torch.ones((n_test, d))) - 1
    y_test = msp.evaluate(X_test)

    def count_matching_rows(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> int:
        """
        Counts the number of rows in tensor_a that exist in tensor_b.

        Args:
            tensor_a (torch.Tensor): Tensor of shape (m, d).
            tensor_b (torch.Tensor): Tensor of shape (n, d).

        Returns:
            int: Number of matching rows.
        """

        # Validate that both tensors are 2D
        if tensor_a.dim() != 2 or tensor_b.dim() != 2:
            raise ValueError("Both tensors must be 2D (matrices).")

        # Validate that both tensors have the same number of columns
        if tensor_a.size(1) != tensor_b.size(1):
            raise ValueError("Both tensors must have the same number of columns (features).")

        # Validate that both tensors have the same data type
        if tensor_a.dtype != tensor_b.dtype:
            raise ValueError("Both tensors must have the same data type.")

        # Move tensors to CPU and convert to NumPy arrays
        a_np = tensor_a.cpu().numpy()
        b_np = tensor_b.cpu().numpy()

        # View each row as a single entity by creating a structured dtype
        # This allows us to treat each row as a unique record
        a_view = a_np.view([('', a_np.dtype)] * a_np.shape[1])
        b_view = b_np.view([('', b_np.dtype)] * b_np.shape[1])

        # Use NumPy's in1d to check for each row in a_view if it exists in b_view
        matches = np.isin(a_view, b_view)

        # Count the number of matches
        count = np.sum(matches)

        return count

    # Iterate over architectures and training sizes
    for n_train in n_train_sizes:
        print(f"\nProcessing n_train = {n_train}")

        # Generate training data
        X_train = 2 * torch.bernoulli(0.5 * torch.ones((n_train, d))) - 1
        y_train = msp.evaluate(X_train)

        print(int(count_matching_rows(X_train, X_test)))

if __name__ == "__main__":
    main()