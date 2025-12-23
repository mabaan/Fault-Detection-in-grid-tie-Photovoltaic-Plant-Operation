# Fault Detection in Grid-Tied Photovoltaic Plant Operation

<div align="center">
  <!-- Add project image/banner here -->
  <img src="" alt="Project Banner" width="800">
</div>

## Overview

This project implements machine learning-based fault detection for grid-connected photovoltaic (PV) systems. By analyzing electrical and environmental parameters from PV strings, the system classifies operational states to identify potential faults that could affect energy production and system reliability.

## Motivation

As solar energy adoption accelerates worldwide, maintaining the efficiency and reliability of PV installations becomes increasingly critical. Faults in PV systems often go undetected for extended periods, leading to significant energy losses, reduced return on investment, and potential safety hazards.

Common issues include:

- Partial shading and soiling effects
- Module degradation and hot spots
- String mismatch and connection failures
- Inverter malfunctions

Traditional monitoring approaches rely on periodic manual inspections, which are time-consuming, costly, and unable to catch intermittent faults. Automated fault detection using machine learning offers a scalable solution that can continuously monitor system performance, identify anomalies in real time, and enable predictive maintenance strategies.

This project explores how supervised learning algorithms can distinguish between normal operation and various fault conditions using readily available sensor data from PV installations.

## Dataset

The input data was engineered from real operational data sourced from a PV plant. The original dataset is available at [clayton-h-costa/pv_fault_dataset](https://github.com/clayton-h-costa/pv_fault_dataset).

### Features

| Feature | Description |
|---------|-------------|
| Voltage - String 1 | Voltage measurement from PV string 1 |
| Voltage - String 2 | Voltage measurement from PV string 2 |
| Current - String 1 | Current measurement from PV string 1 |
| Current - String 2 | Current measurement from PV string 2 |
| Irradiance | Solar irradiance level |
| PV Module Temperature | Temperature of the PV modules |
| Fault Label | Target variable indicating fault condition |

## Methodology

### Data Preprocessing

- Missing value removal
- Feature scaling using MinMax and Standard scaling techniques
- Train-test split with multiple random states for robust evaluation

### Machine Learning Models

The following classifiers are evaluated for fault detection:

- Decision Trees
- k-Nearest Neighbors (k-NN)
- Naive Bayes
- Support Vector Machines (RBF kernel)
- Support Vector Machines (Polynomial kernel)
- Neural Networks (MLP)

### Exploratory Data Analysis

The analysis includes:

- Histograms for feature distribution analysis
- Box plots for outlier detection
- Correlation heatmaps for feature relationship analysis
- Pair plots for multivariate visualization

## Results

The models were evaluated on a 70/30 train-test split. Below are the classification accuracy scores:

| Classifier | Accuracy |
|------------|----------|
| k-Nearest Neighbors | 99.84% |
| Decision Trees | 99.81% |
| Neural Networks (MLP) | 99.55% |
| SVM (Polynomial) | 99.18% |
| SVM (RBF) | 99.03% |
| Naive Bayes | 76.82% |

k-NN and Decision Trees achieved the highest accuracy, both exceeding 99.8%. These results indicate that the selected features provide strong discriminative power for fault classification. The relatively lower performance of Naive Bayes suggests that the independence assumption does not hold well for this dataset, as the electrical parameters are inherently correlated.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository
2. Ensure the dataset file `PV_Data.csv` is in the project directory
3. Run the Jupyter notebook `code.ipynb` to execute the analysis and model training

## Project Structure

```
.
├── code.ipynb          # Main analysis and model training notebook
├── PV_Data.csv         # Processed dataset
├── dataset_amb.mat     # Ambient data (MATLAB format)
├── dataset_elec.mat    # Electrical data (MATLAB format)
├── dataset.ipynb       # Dataset preparation notebook
└── README.md           # Project documentation
```

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

Original PV fault dataset provided by [clayton-h-costa](https://github.com/clayton-h-costa/pv_fault_dataset).
