"""
Experiment: sklearn Benchmarks - GENREG vs Dense Backprop on Real Data

Goal: Find problems where backprop achieves high accuracy with expensive networks,
      and test if GENREG can match accuracy with fewer params.

Datasets:
1. Breast Cancer - 30 features, binary classification, ~95% accuracy
2. Digits - 64 features, 10-class classification, ~97% accuracy
3. California Housing - 8 features, regression, R²~0.8
4. Covtype - 54 features, 7-class classification, ~95% accuracy

For each dataset we compare:
- Dense MLP (backprop): Standard architecture, full connectivity
- Ultra-Sparse GENREG: K=4 inputs per neuron, SA training

Success metric: Similar accuracy with 10-100x fewer params
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from dataclasses import dataclass
from typing import Tuple, List, Dict

from core.models import UltraSparseController, DenseController


@dataclass
class BenchmarkResult:
    dataset: str
    method: str
    metric: float  # accuracy or R²
    metric_name: str
    params: int
    train_time: float


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, str, bool]:
    """Load sklearn dataset. Returns X, y, metric_name, is_classification."""
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, "accuracy", True
    elif name == "digits":
        data = load_digits()
        return data.data, data.target, "accuracy", True
    elif name == "wine":
        data = load_wine()
        return data.data, data.target, "accuracy", True
    else:
        raise ValueError(f"Unknown dataset: {name}")


class DenseClassifier(nn.Module):
    """Dense MLP for classification."""
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class DenseRegressor(nn.Module):
    """Dense MLP for regression."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def train_dense_classifier(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    input_size: int, num_classes: int,
    hidden_size: int = 64,
    epochs: int = 200,
) -> BenchmarkResult:
    """Train dense MLP classifier with backprop."""

    model = DenseClassifier(input_size, hidden_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return BenchmarkResult(
        dataset="", method="Dense Backprop",
        metric=accuracy, metric_name="accuracy",
        params=model.num_parameters(),
        train_time=train_time
    )


def train_dense_regressor(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    input_size: int,
    hidden_size: int = 64,
    epochs: int = 200,
) -> BenchmarkResult:
    """Train dense MLP regressor with backprop."""

    model = DenseRegressor(input_size, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    start = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    train_time = time.time() - start

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        r2 = 1 - ((preds - y_test) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
        r2 = r2.item()

    return BenchmarkResult(
        dataset="", method="Dense Backprop",
        metric=r2, metric_name="R²",
        params=model.num_parameters(),
        train_time=train_time
    )


def train_genreg_classifier(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    input_size: int, num_classes: int,
    hidden_size: int = 8,
    inputs_per_neuron: int = 4,
    max_steps: int = 20000,
) -> BenchmarkResult:
    """Train Ultra-Sparse GENREG classifier with SA."""

    # For classification, we need one output per class
    # Use one-hot encoding for targets
    y_onehot = torch.zeros(len(y_train), num_classes)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8  # Scale to [-0.8, 0.8] for tanh

    controller = UltraSparseController(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        inputs_per_neuron=inputs_per_neuron
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(X_train)
        current_mse = torch.mean((pred - y_onehot) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.1, 0.0001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    start = time.time()
    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(weight_rate=0.15, weight_scale=0.15, index_swap_rate=0.1)

        with torch.no_grad():
            pred = mutant.forward(X_train)
            mutant_mse = torch.mean((pred - y_onehot) ** 2).item()

        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

    train_time = time.time() - start

    # Evaluate
    with torch.no_grad():
        preds = best.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return BenchmarkResult(
        dataset="", method="GENREG (Ultra-Sparse)",
        metric=accuracy, metric_name="accuracy",
        params=best.num_parameters(),
        train_time=train_time
    )


def train_genreg_regressor(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    input_size: int,
    hidden_size: int = 8,
    inputs_per_neuron: int = 4,
    max_steps: int = 20000,
) -> BenchmarkResult:
    """Train Ultra-Sparse GENREG regressor with SA."""

    # Normalize targets for tanh
    y_mean, y_std = y_train.mean(), y_train.std()
    y_norm = (y_train - y_mean) / (y_std + 1e-8) * 0.8

    controller = UltraSparseController(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=1,
        inputs_per_neuron=inputs_per_neuron
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(X_train).squeeze()
        current_mse = torch.mean((pred - y_norm) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.1, 0.0001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    start = time.time()
    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(weight_rate=0.15, weight_scale=0.15, index_swap_rate=0.1)

        with torch.no_grad():
            pred = mutant.forward(X_train).squeeze()
            mutant_mse = torch.mean((pred - y_norm) ** 2).item()

        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

    train_time = time.time() - start

    # Evaluate (denormalize predictions)
    with torch.no_grad():
        pred_norm = best.forward(X_test).squeeze()
        preds = pred_norm / 0.8 * y_std + y_mean
        r2 = 1 - ((preds - y_test) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
        r2 = r2.item()

    return BenchmarkResult(
        dataset="", method="GENREG (Ultra-Sparse)",
        metric=r2, metric_name="R²",
        params=best.num_parameters(),
        train_time=train_time
    )


def run_benchmark(
    dataset_name: str,
    dense_hidden: int = 64,
    genreg_hidden: int = 8,
    genreg_k: int = 4,
    n_trials: int = 3,
    sa_steps: int = 20000,
    bp_epochs: int = 200,
):
    """Run benchmark on one dataset."""

    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    # Load data
    X, y, metric_name, is_classification = load_dataset(dataset_name)

    # Preprocess
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long if is_classification else torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long if is_classification else torch.float32)

    input_size = X_train.shape[1]
    num_classes = len(torch.unique(y_train)) if is_classification else None

    print(f"Samples: {len(X_train)} train, {len(X_test)} test")
    print(f"Features: {input_size}")
    if is_classification:
        print(f"Classes: {num_classes}")
    print(f"Metric: {metric_name}")

    results = {"dense": [], "genreg": []}

    for trial in range(n_trials):
        print(f"\n  Trial {trial + 1}/{n_trials}")
        np.random.seed(trial * 100)
        torch.manual_seed(trial * 100)

        # Dense Backprop
        if is_classification:
            dense_result = train_dense_classifier(
                X_train, y_train, X_test, y_test,
                input_size, num_classes, dense_hidden, bp_epochs
            )
        else:
            dense_result = train_dense_regressor(
                X_train, y_train, X_test, y_test,
                input_size, dense_hidden, bp_epochs
            )
        dense_result.dataset = dataset_name
        results["dense"].append(dense_result)

        # GENREG
        if is_classification:
            genreg_result = train_genreg_classifier(
                X_train, y_train, X_test, y_test,
                input_size, num_classes, genreg_hidden, genreg_k, sa_steps
            )
        else:
            genreg_result = train_genreg_regressor(
                X_train, y_train, X_test, y_test,
                input_size, genreg_hidden, genreg_k, sa_steps
            )
        genreg_result.dataset = dataset_name
        results["genreg"].append(genreg_result)

        print(f"    Dense:  {metric_name}={dense_result.metric:.4f}, params={dense_result.params}")
        print(f"    GENREG: {metric_name}={genreg_result.metric:.4f}, params={genreg_result.params}")

    return results


def run_all_benchmarks():
    """Run benchmarks on all datasets."""

    print("=" * 70)
    print("SKLEARN BENCHMARKS: GENREG vs Dense Backprop")
    print("=" * 70)
    print("\nGoal: Match backprop accuracy with 10-100x fewer params")

    # Different configs for different datasets
    configs = {
        "breast_cancer": {"genreg_hidden": 8, "genreg_k": 4},   # Binary: small network
        "digits": {"genreg_hidden": 32, "genreg_k": 8},          # 10-class: bigger network
        "wine": {"genreg_hidden": 8, "genreg_k": 4},             # 3-class: small network
    }

    datasets = ["breast_cancer", "wine", "digits"]  # Use built-in datasets only

    all_results = {}

    for dataset in datasets:
        cfg = configs[dataset]
        results = run_benchmark(
            dataset,
            dense_hidden=64,
            genreg_hidden=cfg["genreg_hidden"],
            genreg_k=cfg["genreg_k"],
            n_trials=3,
            sa_steps=30000,  # More steps for harder problems
            bp_epochs=200
        )
        all_results[dataset] = results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Dataset':<20} | {'Method':<15} | {'Metric':<12} | {'Params':<8} | {'Efficiency':<10}")
    print("-" * 75)

    for dataset, results in all_results.items():
        dense_metric = np.mean([r.metric for r in results["dense"]])
        genreg_metric = np.mean([r.metric for r in results["genreg"]])

        dense_params = results["dense"][0].params
        genreg_params = results["genreg"][0].params

        metric_name = results["dense"][0].metric_name

        param_ratio = dense_params / genreg_params
        metric_diff = dense_metric - genreg_metric

        print(f"{dataset:<20} | {'Dense':<15} | {dense_metric:<12.4f} | {dense_params:<8} | baseline")
        print(f"{'':<20} | {'GENREG':<15} | {genreg_metric:<12.4f} | {genreg_params:<8} | {param_ratio:.0f}x fewer")
        print(f"{'':<20} | {'Difference':<15} | {metric_diff:+.4f}      |          |")
        print("-" * 75)

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "sklearn_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for dataset, results in all_results.items():
        summary[dataset] = {
            "dense": {
                "metric": np.mean([r.metric for r in results["dense"]]),
                "params": results["dense"][0].params,
            },
            "genreg": {
                "metric": np.mean([r.metric for r in results["genreg"]]),
                "params": results["genreg"][0].params,
            }
        }

    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
