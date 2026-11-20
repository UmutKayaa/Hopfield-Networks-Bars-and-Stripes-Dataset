
import numpy as np


class HopfieldNetwork:
    """
    Hopfield Network for associative memory.

    Uses Hebbian learning and asynchronous neuron updates.
    """

    def __init__(self, num_neurons):
        if num_neurons <= 0:
            raise ValueError("Number of neurons must be positive.")

        self.num_neurons = num_neurons

        self.weights = np.zeros(
            (num_neurons, num_neurons),
            dtype=float
        )

        self.is_trained = False

    def train(self, patterns):
        """
        Train the network using Hebbian learning.
        Patterns must contain bipolar values (-1, 1).
        """

        patterns = np.asarray(patterns)

        if patterns.ndim != 2:
            raise ValueError("Patterns must be a 2D array.")

        if patterns.shape[1] != self.num_neurons:
            raise ValueError("Pattern size does not match network size.")

        if not np.all(np.isin(patterns, [-1, 1])):
            raise ValueError("Patterns must contain only -1 and 1.")

        self.weights.fill(0)

        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)

        self.weights /= self.num_neurons

        # Remove self-connections
        np.fill_diagonal(self.weights, 0)

        self.is_trained = True

    def recall(
        self,
        pattern,
        steps=20,
        random_update=False,
        rng=None
    ):
        """
        Retrieve a stored pattern from a noisy input.
        Uses asynchronous neuron updates.
        """

        if not self.is_trained:
            raise RuntimeError("Train the network before recall.")

        pattern = np.asarray(pattern).copy()

        if pattern.shape != (self.num_neurons,):
            raise ValueError("Invalid input pattern size.")

        if not np.all(np.isin(pattern, [-1, 1])):
            raise ValueError("Input must contain only -1 and 1.")

        rng = rng or np.random.default_rng()

        for _ in range(steps):

            previous_state = pattern.copy()

            if random_update:
                update_order = rng.permutation(self.num_neurons)
            else:
                update_order = range(self.num_neurons)

            for i in update_order:

                weighted_sum = np.dot(
                    self.weights[i],
                    pattern
                )

                if weighted_sum > 0:
                    pattern[i] = 1

                elif weighted_sum < 0:
                    pattern[i] = -1

            if np.array_equal(previous_state, pattern):
                break

        return pattern.copy()
