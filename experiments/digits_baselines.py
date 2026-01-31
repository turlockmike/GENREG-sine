"""
Experiment: Digits Classification Baselines

Question: How does GENREG compare to standard ML models on digits?

Baselines:
- Logistic Regression (linear, ~650 params)
- SVM with RBF kernel (classic for digits)
- MLP with backprop (same-ish architecture as GENREG)
- k-NN (lazy learner, no params)
- Random Forest (ensemble)

This establishes the accuracy ceiling and shows what GENREG is competing against.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def count_params_mlp(mlp):
    """Count parameters in sklearn MLP."""
    total = 0
    for i, coef in enumerate(mlp.coefs_):
        total += coef.size
    for bias in mlp.intercepts_:
        total += bias.size
    return total


def run_baselines():
    """Run all baseline models and report results."""

    # Load data (same split as GENREG experiments)
    data = load_digits()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n{'='*70}")
    print("DIGITS CLASSIFICATION BASELINES")
    print(f"{'='*70}")
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test, 64 features, 10 classes")
    print(f"{'='*70}\n")

    # Define models
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": "64*10 + 10 = 650"  # weights + bias
        },
        "SVM (RBF)": {
            "model": SVC(kernel='rbf', random_state=42),
            "params": "n_support_vectors"
        },
        "MLP (64→64→32→10)": {
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            "params": "computed"
        },
        "MLP (64→32→10) small": {
            "model": MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=500,
                random_state=42,
                early_stopping=True
            ),
            "params": "computed"
        },
        "k-NN (k=3)": {
            "model": KNeighborsClassifier(n_neighbors=3),
            "params": "0 (lazy)"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "params": "~trees*nodes"
        },
    }

    results = []

    for name, config in models.items():
        print(f"Training {name}...", end=" ", flush=True)

        model = config["model"]
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        # Count params
        if "MLP" in name:
            params = count_params_mlp(model)
        elif "SVM" in name:
            params = f"{model.n_support_.sum()} SVs"
        elif "Logistic" in name:
            params = model.coef_.size + model.intercept_.size
        elif "k-NN" in name:
            params = 0
        elif "Random Forest" in name:
            params = sum(tree.tree_.node_count for tree in model.estimators_)
        else:
            params = "?"

        results.append({
            "name": name,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "params": params,
            "train_time": train_time
        })

        print(f"done ({train_time:.2f}s)")

    # Print results table
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Model':<25} | {'Train Acc':<10} | {'Test Acc':<10} | {'Params':<12} | {'Time':<8}")
    print("-" * 70)

    for r in results:
        params_str = str(r['params']) if isinstance(r['params'], int) else r['params']
        print(f"{r['name']:<25} | {r['train_acc']:<10.1%} | {r['test_acc']:<10.1%} | {params_str:<12} | {r['train_time']:.2f}s")

    print("-" * 70)

    # Add GENREG comparison
    print(f"\n{'='*70}")
    print("COMPARISON WITH GENREG")
    print(f"{'='*70}")
    print("""
GENREG GSA Results (from experiments):
  - H=32, K=4:  ~84% test acc, 490 params (current run, gen ~1000)
  - H=64, K=16: 87.2% test acc, 1738 params (previous best)

Key Observations:
  - Best baseline (SVM/MLP): 97-98% accuracy
  - GENREG gap: ~10-14 percentage points below best
  - GENREG advantage: 3-5x fewer parameters than comparable MLP
  - GENREG trains slower but produces sparser models
""")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable
    json_results = []
    for r in results:
        jr = r.copy()
        jr['params'] = str(r['params'])
        json_results.append(jr)

    with open(output_dir / "digits_baselines.json", "w") as f:
        json.dump({
            "dataset": "sklearn.digits",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "results": json_results
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}/digits_baselines.json")

    return results


if __name__ == "__main__":
    run_baselines()
