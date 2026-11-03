# K-Nearest Neighbors Classifier

A machine learning project that implements a K-Nearest Neighbors (KNN) classifier to predict customer categories for a telecommunications company based on demographic and behavioral data.

## Overview

This project demonstrates the full workflow of building a K-Nearest Neighbors classification model using Python and scikit-learn. It covers data loading, exploratory data analysis, feature correlation analysis, data normalization, model training, and evaluation on a customer segmentation dataset.

## Dataset

The project uses the **teleCust1000t** dataset, a publicly available telecommunications customer dataset containing 1,000 records. Each record describes a customer through the following features:

| Feature | Description |
|---------|-------------|
| region | Geographic region of the customer |
| tenure | Number of years as a customer |
| age | Customer age |
| marital | Marital status (binary encoded) |
| address | Number of years at current address |
| income | Household income (in thousands) |
| ed | Education level |
| employ | Years employed |
| retire | Retirement status |
| gender | Gender (binary encoded) |
| reside | Number of people in household |
| custcat | Customer category (target variable, 1 to 4) |

The target variable, `custcat`, consists of four customer categories:

- Category 1: 266 records
- Category 2: 217 records
- Category 3: 281 records
- Category 4: 236 records

## Methodology

### 1. Data Loading and Inspection

The dataset is loaded directly from a remote CSV source using pandas. An initial inspection of the first five records and the distribution of the target variable is performed to understand the data structure.

### 2. Exploratory Data Analysis

A correlation heatmap is generated using seaborn and matplotlib to visualize relationships between all numerical features. This is followed by a ranked analysis of feature correlations with respect to the target variable.

<img width="469" height="283" alt="image" src="https://github.com/user-attachments/assets/8b29aa13-d626-4b8e-8504-9fad243d29e3" />


The correlation analysis reveals that education (`ed`, 0.194) and tenure (`tenure`, 0.167) have the strongest positive relationships with the customer category, while `gender` (0.005) and `retire` (0.009) show negligible influence.

### 3. Feature Preparation

The dataset is split into input features (`X`) and the target label (`y`). All input features are then standardized using `StandardScaler`. Normalization is critical for KNN because the algorithm relies on distance calculations, and features with larger numeric ranges would otherwise dominate the results.

### 4. Train-Test Split

The normalized data is divided into training (80 percent) and testing (20 percent) subsets using a fixed random state of 4 to ensure reproducibility.

### 5. Model Training

A `KNeighborsClassifier` from scikit-learn is instantiated with `k = 3` neighbors and fitted on the training data.

### 6. Evaluation

The trained model is used to predict customer categories for the test set, and performance is measured using the accuracy classification score, which computes subset accuracy for multi-class classification.

## Results

| Metric | Value |
|--------|-------|
| Number of neighbors (k) | 3 |
| Test set size | 20 percent |
| Test set accuracy | 31.5 percent |

The initial accuracy of 31.5 percent indicates room for improvement. This result is expected given the relatively weak correlation between individual features and the target variable, and suggests opportunities for tuning the model further.

## Requirements

The project depends on the following Python libraries:

```
numpy==2.2.0
pandas==2.2.3
scikit-learn==1.6.0
matplotlib==3.9.3
seaborn==0.13.2
```

Python 3.13 or later is recommended.

## Getting Started

### Installation

Install the required dependencies using pip:

```bash
pip install numpy==2.2.0 pandas==2.2.3 scikit-learn==1.6.0 matplotlib==3.9.3 seaborn==0.13.2
```

### Running the Notebook

1. Clone the repository:

```bash
git clone https://github.com/MawiraVictor/K-Nearest-Neighbors-Classifier.git
cd K-Nearest-Neighbors-Classifier
```

2. Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

3. Open `K-Nearest Neighbors Classifier.ipynb` and run the cells in sequence.

The dataset is fetched automatically from its remote URL, so no local data files are required.

## Project Structure

```
K-Nearest-Neighbors-Classifier/
|
|-- K-Nearest Neighbors Classifier.ipynb
|-- README.md
```

## Potential Improvements

The following enhancements could be explored to improve model performance and robustness:

- Hyperparameter tuning: evaluate multiple values of `k` using cross-validation to identify the optimal number of neighbors.
- Feature engineering: derive new features or apply dimensionality reduction techniques such as principal component analysis.
- Distance metrics: experiment with alternative distance measures such as Manhattan or Minkowski distance.
- Class balancing: investigate resampling strategies to address the mild class imbalance in the target variable.

## License

This project is provided for educational purposes. Refer to the repository for licensing details.

## Author

**Mawira Victor**

GitHub: [https://github.com/MawiraVictor](https://github.com/MawiraVictor)

Repository: [https://github.com/MawiraVictor/K-Nearest-Neighbors-Classifier](https://github.com/MawiraVictor/K-Nearest-Neighbors-Classifier)
