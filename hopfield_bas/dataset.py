
import numpy as np
from itertools import combinations


def generate_all_bas_patterns(size=8):
    """
    Generate all unique Bars and Stripes patterns.

    Bars: Vertical patterns.
    Stripes: Horizontal patterns.
    """

    unique_patterns = {}

    # Generate vertical bars
    for num_bars in range(1, size + 1):
        for indices in combinations(range(size), num_bars):

            sample = np.zeros((size, size), dtype=int)
            sample[:, indices] = 1

            key = tuple(sample.flatten())
            unique_patterns[key] = sample

    # Generate horizontal stripes
    for num_stripes in range(1, size + 1):
        for indices in combinations(range(size), num_stripes):

            sample = np.zeros((size, size), dtype=int)
            sample[indices, :] = 1

            key = tuple(sample.flatten())
            unique_patterns[key] = sample

    return list(unique_patterns.values())


def get_unique_bas_samples(num_samples=10, size=8, rng=None):
    """Select unique BAS samples."""

    all_patterns = generate_all_bas_patterns(size)

    if num_samples > len(all_patterns):
        raise ValueError(
            f"Only {len(all_patterns)} unique patterns available."
        )

    if rng is None:
        selected_indices = np.arange(num_samples)
    else:
        selected_indices = rng.choice(
            len(all_patterns),
            size=num_samples,
            replace=False
        )

    return [all_patterns[i] for i in selected_indices]


def to_bipolar(pattern):
    """Convert binary patterns (0, 1) to bipolar (-1, 1)."""

    return np.asarray(pattern).flatten() * 2 - 1


def add_noise(pattern, noise_level=0.1, rng=None):
    """Add noise by randomly flipping bits."""

    if not 0 <= noise_level <= 1:
        raise ValueError("Noise level must be between 0 and 1.")

    noisy_pattern = np.asarray(pattern).copy()

    if not np.all(np.isin(noisy_pattern, [-1, 1])):
        raise ValueError("Pattern must contain -1 and 1.")

    rng = rng or np.random.default_rng()

    num_noisy_bits = int(round(noise_level * noisy_pattern.size))

    if num_noisy_bits == 0:
        return noisy_pattern

    flip_indices = rng.choice(
        noisy_pattern.size,
        size=num_noisy_bits,
        replace=False
    )

    noisy_pattern[flip_indices] *= -1

    return noisy_pattern
