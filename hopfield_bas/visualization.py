
import numpy as np
import matplotlib.pyplot as plt


def visualize_bas_dataset(dataset):
    """Visualize Bars and Stripes dataset."""

    num_samples = len(dataset)

    if num_samples == 0:
        raise ValueError("Dataset cannot be empty.")

    fig, axes = plt.subplots(
        1,
        num_samples,
        figsize=(num_samples * 2, 2)
    )

    if num_samples == 1:
        axes = [axes]

    for ax, sample in zip(axes, dataset):
        ax.imshow(
            sample,
            cmap="gray",
            interpolation="nearest"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_recall_result(result, size=8):
    """Compare noisy input, recalled output and original pattern."""

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    images = [
        (result["noisy"], "Noisy Input"),
        (result["recalled"], "Recalled Output"),
        (result["original"], "Original Pattern")
    ]

    for ax, (pattern, title) in zip(axes, images):

        ax.imshow(
            np.asarray(pattern).reshape(size, size),
            cmap="gray",
            interpolation="nearest",
            vmin=-1,
            vmax=1
        )

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_noise(results, title=None):
    """Plot retrieval accuracy against noise levels."""

    noise_levels = list(results.keys())

    accuracies = [
        results[level]["accuracy"]
        for level in noise_levels
    ]

    exact_rates = [
        results[level]["exact_retrieval_rate"]
        for level in noise_levels
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        noise_levels,
        accuracies,
        marker="o",
        label="Bitwise Accuracy"
    )

    plt.plot(
        noise_levels,
        exact_rates,
        marker="s",
        linestyle="--",
        label="Exact Retrieval Rate"
    )

    plt.xlabel("Noise Level")
    plt.ylabel("Performance")

    plt.title(
        title or "Retrieval Performance vs Noise"
    )

    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_capacity_results(results):
    """Plot retrieval performance against memory load."""

    sample_sizes = list(results.keys())

    mean_errors = [
        results[n]["mean_errors"]
        for n in sample_sizes
    ]

    exact_rates = [
        results[n]["exact_retrieval_rate"]
        for n in sample_sizes
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(
        sample_sizes,
        mean_errors,
        marker="o",
        color="red"
    )

    axes[0].set_title("Mean Retrieval Errors")
    axes[0].set_xlabel("Number of Stored Patterns")
    axes[0].set_ylabel("Mean Number of Errors")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        sample_sizes,
        exact_rates,
        marker="o",
        color="green"
    )

    axes[1].set_title("Exact Retrieval Rate")
    axes[1].set_xlabel("Number of Stored Patterns")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_error_distribution(
    error_results,
    size=8,
    title="Retrieval Error Distribution"
):
    """Plot probability distribution of retrieval errors."""

    bins = np.arange(0, size * size + 2) - 0.5

    plt.figure(figsize=(8, 5))

    plt.hist(
        error_results,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.75,
        hatch="//"
    )

    plt.xlabel("Number of Incorrect Bits")
    plt.ylabel("Probability")
    plt.title(title)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
