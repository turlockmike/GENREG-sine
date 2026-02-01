"""
Data augmentation utilities for GENREG experiments.

Supports sklearn digits (8x8) and other image datasets.
"""

import numpy as np
from scipy.ndimage import rotate, shift


def augment_digits(X, y=None, rotation_range=10, shift_range=1, n_augmented=1, seed=None):
    """
    Augment sklearn digits dataset with rotations and shifts.

    Args:
        X: Input data, shape (n_samples, 64) for digits or (n_samples, features)
        y: Labels (optional, will be duplicated for augmented samples)
        rotation_range: Max rotation in degrees (±rotation_range)
        shift_range: Max shift in pixels (±shift_range)
        n_augmented: Number of augmented copies per original sample
        seed: Random seed for reproducibility

    Returns:
        X_aug: Augmented data (original + augmented samples)
        y_aug: Labels for augmented data (if y provided)
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, n_features = X.shape

    # Determine image size (assume square)
    img_size = int(np.sqrt(n_features))
    if img_size * img_size != n_features:
        raise ValueError(f"Features ({n_features}) must be a perfect square for image augmentation")

    augmented_X = [X]  # Start with original
    augmented_y = [y] if y is not None else None

    for _ in range(n_augmented):
        X_new = np.zeros_like(X)

        for i in range(n_samples):
            # Reshape to image
            img = X[i].reshape(img_size, img_size)

            # Random rotation
            if rotation_range > 0:
                angle = np.random.uniform(-rotation_range, rotation_range)
                img = rotate(img, angle, reshape=False, mode='constant', cval=0)

            # Random shift
            if shift_range > 0:
                shift_y = np.random.uniform(-shift_range, shift_range)
                shift_x = np.random.uniform(-shift_range, shift_range)
                img = shift(img, [shift_y, shift_x], mode='constant', cval=0)

            # Flatten back
            X_new[i] = img.flatten()

        augmented_X.append(X_new)
        if y is not None:
            augmented_y.append(y.copy())

    X_aug = np.vstack(augmented_X)

    if y is not None:
        y_aug = np.concatenate(augmented_y)
        return X_aug, y_aug

    return X_aug


def augment_batch(X_batch, rotation_range=10, shift_range=1):
    """
    Apply random augmentation to a batch (for on-the-fly augmentation).

    Args:
        X_batch: Batch of samples, shape (batch_size, 64)
        rotation_range: Max rotation in degrees
        shift_range: Max shift in pixels

    Returns:
        X_aug: Augmented batch (same shape as input)
    """
    batch_size, n_features = X_batch.shape
    img_size = int(np.sqrt(n_features))

    X_aug = np.zeros_like(X_batch)

    for i in range(batch_size):
        img = X_batch[i].reshape(img_size, img_size)

        # Random rotation
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            img = rotate(img, angle, reshape=False, mode='constant', cval=0)

        # Random shift
        if shift_range > 0:
            shift_y = np.random.uniform(-shift_range, shift_range)
            shift_x = np.random.uniform(-shift_range, shift_range)
            img = shift(img, [shift_y, shift_x], mode='constant', cval=0)

        X_aug[i] = img.flatten()

    return X_aug


def visualize_augmentation(X_sample, rotation_range=10, shift_range=1, n_examples=5):
    """
    Visualize augmentation on a single sample (for debugging).

    Args:
        X_sample: Single sample, shape (64,) for digits
        rotation_range: Max rotation
        shift_range: Max shift
        n_examples: Number of augmented examples to show

    Returns:
        List of (original, augmented) image pairs as 8x8 arrays
    """
    img_size = int(np.sqrt(len(X_sample)))
    original = X_sample.reshape(img_size, img_size)

    examples = [(original, "Original")]

    for i in range(n_examples):
        img = original.copy()

        angle = np.random.uniform(-rotation_range, rotation_range)
        img = rotate(img, angle, reshape=False, mode='constant', cval=0)

        if shift_range > 0:
            sy = np.random.uniform(-shift_range, shift_range)
            sx = np.random.uniform(-shift_range, shift_range)
            img = shift(img, [sy, sx], mode='constant', cval=0)

        examples.append((img, f"rot={angle:.1f}°"))

    return examples
