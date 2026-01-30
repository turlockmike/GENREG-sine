"""
Quick tests to verify USEN package works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    from usen import (
        SparseNet, DenseNet,
        train_gsa, train_sa,
        sine_problem, highdim_problem,
        mse, saturation, selection_stats
    )
    print("  All imports OK")


def test_sparse_net():
    """Test SparseNet basic functionality."""
    print("Testing SparseNet...")
    from usen import SparseNet

    net = SparseNet(input_dim=100, H=8, K=4)
    assert net.num_params() == 8*4 + 8 + 8 + 1  # 49

    X = np.random.randn(50, 100).astype(np.float32)
    h, out = net.forward(X)
    assert h.shape == (50, 8)
    assert out.shape == (50,)
    assert np.all(np.abs(out) <= 1)  # tanh output

    # Test clone
    net2 = net.clone()
    out2 = net2.predict(X)
    assert np.allclose(out, out2)

    # Test mutate
    net2.mutate()
    out3 = net2.predict(X)
    assert not np.allclose(out, out3)  # Should be different after mutation

    print(f"  SparseNet OK (params={net.num_params()})")


def test_sine_problem():
    """Test sine problem generation."""
    print("Testing sine_problem...")
    from usen import sine_problem

    X, y, true_features = sine_problem(n_samples=100, n_features=16)
    assert X.shape == (100, 16)
    assert y.shape == (100,)
    assert len(true_features) == 16

    print(f"  sine_problem OK (X={X.shape}, y={y.shape})")


def test_highdim_problem():
    """Test high-dim problem generation."""
    print("Testing highdim_problem...")
    from usen import highdim_problem

    X, y, true_features = highdim_problem(n_samples=100, n_features=1000, n_true=10)
    assert X.shape == (100, 1000)
    assert y.shape == (100,)
    assert true_features == list(range(10))

    print(f"  highdim_problem OK (X={X.shape}, true={len(true_features)})")


def test_train_gsa_quick():
    """Quick test of GSA training (minimal iterations)."""
    print("Testing train_gsa (quick)...")
    from usen import SparseNet, train_gsa, sine_problem, mse

    X, y, _ = sine_problem(n_samples=100, n_features=16)
    net = SparseNet(input_dim=16, H=4, K=2)

    initial_mse = mse(net, X, y)

    # Very short training just to test it runs
    best, history = train_gsa(
        net, X, y,
        generations=5,
        pop_size=10,
        sa_steps=5,
        seed=42
    )

    final_mse = mse(best, X, y)
    assert final_mse < initial_mse  # Should improve

    print(f"  train_gsa OK (MSE: {initial_mse:.4f} -> {final_mse:.4f})")


def test_train_sa_quick():
    """Quick test of SA training."""
    print("Testing train_sa (quick)...")
    from usen import SparseNet, train_sa, sine_problem, mse

    X, y, _ = sine_problem(n_samples=100, n_features=16)
    net = SparseNet(input_dim=16, H=4, K=2)

    initial_mse = mse(net, X, y)

    best, history = train_sa(
        net, X, y,
        max_steps=500,
        seed=42
    )

    final_mse = mse(best, X, y)
    assert final_mse < initial_mse

    print(f"  train_sa OK (MSE: {initial_mse:.4f} -> {final_mse:.4f})")


def test_metrics():
    """Test metrics functions."""
    print("Testing metrics...")
    from usen import SparseNet, sine_problem, mse, saturation, selection_stats

    X, y, true_features = sine_problem(n_samples=100, n_features=16)
    net = SparseNet(input_dim=16, H=4, K=2)

    m = mse(net, X, y)
    assert 0 <= m <= 2  # Reasonable MSE range

    s = saturation(net, X)
    assert 0 <= s <= 1

    stats = selection_stats(net, true_features)
    assert 'selection_factor' in stats
    assert stats['total_connections'] == 4 * 2  # H * K

    print(f"  metrics OK (MSE={m:.4f}, sat={s:.2%})")


def test_full_workflow():
    """Test complete training workflow."""
    print("Testing full workflow...")
    from usen import SparseNet, train_gsa, highdim_problem, mse, selection_stats

    # Generate problem
    X, y, true_features = highdim_problem(n_samples=200, n_features=100, n_true=5)

    # Create and train network
    net = SparseNet(input_dim=100, H=8, K=4)
    best, history = train_gsa(
        net, X, y,
        generations=10,
        pop_size=20,
        seed=42
    )

    # Evaluate
    final_mse = mse(best, X, y)
    stats = selection_stats(best, true_features)

    print(f"  Full workflow OK")
    print(f"    MSE: {final_mse:.4f}")
    print(f"    True found: {stats['true_features_found']}/{stats['true_features_total']}")
    print(f"    Selection factor: {stats['selection_factor']:.1f}x")


def main():
    print("=" * 60)
    print("USEN Package Tests")
    print("=" * 60)

    test_imports()
    test_sparse_net()
    test_sine_problem()
    test_highdim_problem()
    test_train_gsa_quick()
    test_train_sa_quick()
    test_metrics()
    test_full_workflow()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
