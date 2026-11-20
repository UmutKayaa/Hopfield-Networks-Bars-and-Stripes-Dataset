# Hopfield Network – Bars and Stripes Dataset

## Project Overview

This project implements a **Hopfield Network** using the Bars and Stripes (BAS) dataset to investigate associative memory, pattern retrieval, noise robustness, storage capacity, and retrieval errors.

The experiments evaluate how effectively a Hopfield Network can store and retrieve binary patterns under different noise levels and memory loads.

## Project Structure

```text
Hopfield-Networks-Bars-and-Stripes-Dataset/
│
├── main.py
│
├── hopfield_bas/
│   ├── dataset.py
│   ├── experiments.py
│   └── visualization.py
│
├── results/
│   ├── figure_1.png
│   ├── figure_2.png
│   ├── figure_3.png
│   ├── figure_4.png
│   ├── figure_5.png
│   ├── figure_6.png
│   └── figure_7.png
│
├── requirements.txt
└── README.md
```

## Experimental Configuration

The experiments use the following parameters:

| Parameter | Value |
|---|---:|
| Pattern Size | 8 × 8 |
| Number of Samples | 10 |
| Number of Trials | 100 |
| Random Seed | 42 |
| Single Recall Noise Level | 0.5 |
| Capacity Experiment Noise | 0.3 |
| Error Distribution Noise | 0.3 |

## Experimental Results

### Figure 1 – Bars and Stripes Dataset Visualization

Displays the generated Bars and Stripes patterns used as the training dataset for the Hopfield Network.

![Figure 1](results/figure_1.png)

### Figure 2 – Single Pattern Recall

Demonstrates the retrieval performance of the Hopfield Network when recalling a noisy input pattern.

The experiment uses a noise level of 0.5 and evaluates:
- Retrieval accuracy
- Number of errors
- Exact pattern retrieval

![Figure 2](results/figure_2.png)

### Figure 3 – Accuracy vs Noise

Evaluates how increasing noise levels affect the retrieval accuracy of the Hopfield Network.

Noise levels range from 0.0 to 0.5, with increments of 0.05.

![Figure 3](results/figure_3.png)

### Figure 4 – Network Storage Capacity

Investigates how the number of stored patterns influences the network's retrieval performance.

The experiment considers the following sample sizes:

- 3 patterns
- 6 patterns
- 9 patterns
- 15 patterns
- 20 patterns

![Figure 4](results/figure_4.png)

### Figure 5 – Error Distribution (3 Stored Patterns)

Visualizes the distribution of retrieval errors when the network stores 3 patterns.

![Figure 5](results/figure_5.png)

### Figure 6 – Error Distribution (6 Stored Patterns)

Visualizes the distribution of retrieval errors when the network stores 6 patterns.

![Figure 6](results/figure_6.png)

### Figure 7 – Error Distribution (9 Stored Patterns)

Visualizes the distribution of retrieval errors when the network stores 9 patterns.

![Figure 7](results/figure_7.png)

## Technologies Used

- Python
- NumPy
- Matplotlib
- Hopfield Neural Networks
- Associative Memory
- Bars and Stripes Dataset

## Installation

Clone the repository:

```bash
git clone https://github.com/UmutKayaa/Hopfield-Networks-Bars-and-Stripes-Dataset.git
```

Navigate to the project directory:

```bash
cd Hopfield-Networks-Bars-and-Stripes-Dataset
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the experiments:

```bash
python main.py
```

## Reproducibility

A fixed random seed (`42`) is used to improve reproducibility across experiments.

Each experiment performs 100 trials unless otherwise specified.

## Conclusion

This project explores the capabilities and limitations of Hopfield Networks as associative memory systems.

Through the Bars and Stripes dataset, the experiments investigate pattern reconstruction, noise tolerance, storage capacity, and retrieval error distributions.
