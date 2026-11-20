
import numpy as np
print("MAIN.PY STARTED")

from hopfield_bas.dataset import (
    get_unique_bas_samples
)

from hopfield_bas.experiments import (
    run_single_recall_experiment,
    run_accuracy_experiment,
    run_capacity_experiment,
    run_error_experiment
)

from hopfield_bas.visualization import (
    visualize_bas_dataset,
    visualize_recall_result,
    plot_accuracy_vs_noise,
    plot_capacity_results,
    plot_error_distribution
)

def main():

    SIZE = 8
    NUM_SAMPLES = 10
    NUM_TRIALS = 100
    SEED = 42

    # ---------------------------------
    # 1. Dataset Visualization
    # ---------------------------------

    dataset = get_unique_bas_samples(
        NUM_SAMPLES,
        SIZE,
        np.random.default_rng(SEED)
    )

    visualize_bas_dataset(dataset)

    # ---------------------------------
    # 2. Single Pattern Recall
    # ---------------------------------

    recall_result = run_single_recall_experiment(
        num_samples=NUM_SAMPLES,
        size=SIZE,
        noise_level=0.5,
        seed=SEED
    )

    print("\nSingle Recall Experiment")
    print("------------------------")
    print("Accuracy:", recall_result["accuracy"])
    print("Errors:", recall_result["errors"])
    print("Exact Retrieval:", recall_result["exact_match"])

    visualize_recall_result(
        recall_result,
        SIZE
    )

    # ---------------------------------
    # 3. Accuracy vs Noise
    # ---------------------------------

    noise_levels = np.arange(0, 0.55, 0.05)

    accuracy_results = run_accuracy_experiment(
        num_samples=NUM_SAMPLES,
        size=SIZE,
        num_trials=NUM_TRIALS,
        noise_levels=noise_levels,
        seed=SEED
    )

    plot_accuracy_vs_noise(
        accuracy_results,
        title="Hopfield Network: Accuracy vs Noise"
    )

    # ---------------------------------
    # 4. Capacity Experiment
    # ---------------------------------

    sample_sizes = [3, 6, 9, 15, 20]

    capacity_results = run_capacity_experiment(
        sample_sizes=sample_sizes,
        size=SIZE,
        num_trials=NUM_TRIALS,
        noise_level=0.3,
        seed=SEED
    )

    plot_capacity_results(capacity_results)

    # ---------------------------------
    # 5. Error Distribution
    # ---------------------------------

    for num_samples in [3, 6, 9]:

        errors = run_error_experiment(
            num_samples=num_samples,
            size=SIZE,
            num_trials=NUM_TRIALS,
            noise_level=0.3,
            seed=SEED
        )

        plot_error_distribution(
            errors,
            size=SIZE,
            title=f"Error Distribution (n={num_samples})"
        )


if __name__ == "__main__":
    main()
