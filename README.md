# KNN-from-Scratch-Curse-of-Dimensionality-Analysis
A modular Python implementation of the K-Nearest Neighbors (KNN) algorithm built from scratch. Features include interactive class prediction, 2D cluster visualization, decision boundary mapping, and an analysis of the Curse of Dimensionality.

# KNN from Scratch & Curse of Dimensionality Analysis

##  Overview
This project provides a modular, from-scratch implementation of the **K-Nearest Neighbors (KNN)** machine learning algorithm using Python. It separates the mathematical logic from the application layer, allowing for interactive predictions and detailed visualizations.

Additionally, the project includes a mathematical simulation demonstrating the **Curse of Dimensionality**, showing how Euclidean distance metrics degrade in high-dimensional spaces.

##  Project Structure

### Core Logic
* **`knn_core.py`**: The library file containing the reusable `knn()` algorithm and `load_data_from_csv()` function. This is imported by the other scripts.

### Interactive Tools
* **`knn_predict.py`**: A command-line interface (CLI) that loads the data, asks for X/Y coordinates, and predicts the class of the new point.
* **`knn_plot.py`**: Visualizes the training dataset and a specific query point on a 2D scatter plot.
* **`knn_boundary.py`**: Generates a decision boundary map, coloring the 2D plane to visualize the regions belonging to each class.

### Analysis
* **`knn_vis.py`**: A simulation script that generates random data in low (2D) vs. high (3000D) dimensions to visualize the "Curse of Dimensionality" via distance histograms.
* **`KNN.ipynb`**: The original Jupyter Notebook containing the step-by-step development, experiments, and theoretical explanations.

### Data
* **`data.csv`**: The dataset used for training the model (Format: `Feature1, Feature2, Class`).

## 🛠️ Installation & Requirements

Ensure you have Python installed. You will need the following libraries:

```bash
pip install -r requirements.txt
