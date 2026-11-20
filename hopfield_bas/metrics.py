
import numpy as np


def hamming_distance(vec1, vec2):
    """
    Calculate Hamming distance between two vectors.
    Returns the number of different bits.
    """

    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape.")

    return int(np.sum(vec1 != vec2))


def retrieval_accuracy(original, recalled):
    """
    Calculate bitwise retrieval accuracy.
    Returns a value between 0 and 1.
    """

    original = np.asarray(original)
    recalled = np.asarray(recalled)

    if original.shape != recalled.shape:
        raise ValueError("Vectors must have the same shape.")

    return float(np.mean(original == recalled))


def find_errors(original, recalled):
    """
    Count incorrectly retrieved bits.
    """

    return hamming_distance(original, recalled)


def exact_retrieval(original, recalled):
    """
    Check whether the entire original pattern
    has been retrieved correctly.
    """

    return bool(np.array_equal(original, recalled))


def find_closest_pattern(pattern, stored_patterns):
    """
    Find the closest stored pattern using
    Hamming distance.
    """

    if len(stored_patterns) == 0:
        raise ValueError("Stored patterns cannot be empty.")

    distances = [
        hamming_distance(pattern, stored)
        for stored in stored_patterns
    ]

    closest_index = int(np.argmin(distances))

    return (
        stored_patterns[closest_index],
        distances[closest_index]
    )
