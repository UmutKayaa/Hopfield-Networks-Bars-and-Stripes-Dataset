
import numpy as np

from .network import HopfieldNetwork
from .dataset import (
    generate_all_bas_patterns,
    get_unique_bas_samples,
    to_bipolar,
    add_noise
)
from .metrics import (
    retrieval_accuracy,
    find_errors,
    exact_retrieval
)


def prepare_network(num_samples, size, rng):
    """Generate training patterns and train Hopfield Network."""

    dataset = get_unique_bas_samples(num_samples, size, rng)

    patterns = np.array([
        to_bipolar(sample) for sample in dataset
    ])

    network = HopfieldNetwork(size * size)
    network.train(patterns)

    return network, patterns


def run_single_recall_experiment(
    num_samples=10,
    size=8,
    noise_level=0.5,
    seed=42
):
    """Run a single pattern retrieval experiment."""

    rng = np.random.default_rng(seed)

    network, patterns = prepare_network(
        num_samples, size, rng
    )

    index = rng.integers(num_samples)
    original = patterns[index]

    noisy = add_noise(original, noise_level, rng)
    recalled = network.recall(noisy)

    return {
        "original": original,
        "noisy": noisy,
        "recalled": recalled,
        "accuracy": retrieval_accuracy(original, recalled),
        "errors": find_errors(original, recalled),
        "exact_match": exact_retrieval(original, recalled)
    }


def run_accuracy_experiment(
    num_samples,
    size=8,
    num_trials=100,
    noise_levels=None,
    seed=42
):
    """Evaluate retrieval accuracy at different noise levels."""

    rng = np.random.default_rng(seed)

    network, patterns = prepare_network(
        num_samples, size, rng
    )

    if noise_levels is None:
        noise_levels = np.arange(0, 0.55, 0.05)

    results = {}

    for noise in noise_levels:

        accuracies = []
        exact_matches = 0

        for _ in range(num_trials):

            index = rng.integers(num_samples)
            original = patterns[index]

            noisy = add_noise(original, noise, rng)
            recalled = network.recall(noisy)

            accuracies.append(
                retrieval_accuracy(original, recalled)
            )

            if exact_retrieval(original, recalled):
                exact_matches += 1

        results[float(noise)] = {
            "accuracy": float(np.mean(accuracies)),
            "exact_retrieval_rate": exact_matches / num_trials
        }

    return results


def run_error_experiment(
    num_samples,
    size=8,
    num_trials=100,
    noise_level=0.3,
    seed=42
):
    """Calculate retrieval errors over multiple trials."""

    rng = np.random.default_rng(seed)

    network, patterns = prepare_network(
        num_samples, size, rng
    )

    error_counts = []

    for _ in range(num_trials):

        index = rng.integers(num_samples)
        original = patterns[index]

        noisy = add_noise(original, noise_level, rng)
        recalled = network.recall(noisy)

        error_counts.append(
            find_errors(original, recalled)
        )

    return error_counts


def run_capacity_experiment(
    sample_sizes,
    size=8,
    num_trials=100,
    noise_level=0.3,
    seed=42
):
    """Analyze performance under different memory loads."""

    results = {}

    for i, num_samples in enumerate(sample_sizes):

        errors = run_error_experiment(
            num_samples=num_samples,
            size=size,
            num_trials=num_trials,
            noise_level=noise_level,
            seed=seed + i
        )

        results[num_samples] = {
            "mean_errors": float(np.mean(errors)),
            "std_errors": float(np.std(errors)),
            "exact_retrieval_rate": float(
                np.mean(np.array(errors) == 0)
            )
        }

    return results


def run_unseen_pattern_experiment(
    num_train_samples=5,
    size=8,
    seed=42
):
    """Test recall using a pattern excluded from training."""

    rng = np.random.default_rng(seed)

    all_patterns = generate_all_bas_patterns(size)

    if num_train_samples >= len(all_patterns):
        raise ValueError("Not enough patterns for unseen testing.")

    indices = rng.choice(
        len(all_patterns),
        size=num_train_samples + 1,
        replace=False
    )

    train_patterns = np.array([
        to_bipolar(all_patterns[i])
        for i in indices[:-1]
    ])

    original = to_bipolar(all_patterns[indices[-1]])

    network = HopfieldNetwork(size * size)
    network.train(train_patterns)

    recalled = network.recall(original)

    return {
        "original": original,
        "recalled": recalled,
        "accuracy": retrieval_accuracy(original, recalled),
        "errors": find_errors(original, recalled)
    }
