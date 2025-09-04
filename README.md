## 4-step Decomposition for Hybrid Liabilities
This repository contains the Python code to reproduce all numerical results and figures from Ling et al. (2025), "A decomposition framework for managing hybrid liabilities".

## Abstract
In this paper, we propose a four-step decomposition of hybrid liabilities into: a hedgeable part, an idiosyncratic part, a financial systematic part, and an actuarial systematic part. Our model generalizes existing approaches by incorporating dependence between financial and actuarial markets and allowing for heterogeneity in policyholder-specific risks. We illustrate the framework using a portfolio of with-profit pure endowment contracts.

## Reproducing the Results
This code reproduces the numerical results from Section 3.4 and Section 4.4 of the paper.

## Prerequisites
Before running the code, ensure you have the following Python packages installed: numpy; scipy; pandas; matplotlib; seaborn (for improved visualizations)

## Output

Running the code will:

Decompose the aggregate liability into the four specified parts.

Calculate and display key statistical properties for each component (e.g., mean, variance, skewness, kurtosis).

Generate plots comparing the distributions of the different parts.

Output the results for the three market-consistent valuations:

MVSD (Mean-Variance Hedge-Based Standard Deviation Principle)

TSSD (Two-Step Standard Deviation Principle)

CMC (Conic Market-Consistent Standard Deviation Principle)

The results will be saved in the output/ directory, including:
figures/: A directory containing all generated plots (e.g., .pdf files).

## Reference
Ling, B., Linders, D., Dhaene, J., and Boonen, T. (2025). "A decomposition framework for managing hybrid liabilities". Working Paper.
