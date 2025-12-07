# FILE: knn_boundary.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn_core import knn, load_data_from_csv

X_train, y_train = load_data_from_csv()
k_value = 3

if X_train is not None:
    print("Calculating boundaries... Please wait.")
    
    # 1. Create a background grid
    h = 0.1  # Mesh step size
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 2. Predict every point in the grid
    unique_classes = np.unique(y_train)
    class_map = {c: i for i, c in enumerate(unique_classes)}
    
    Z = []
    # Loop over the grid (this can be slow)
    for i in range(xx.shape[0]):
        row = []
        for j in range(xx.shape[1]):
            pt = np.array([xx[i, j], yy[i, j]])
            pred = knn((X_train, y_train), pt, k_value)
            row.append(class_map[pred])
        Z.append(row)
    Z = np.array(Z)

    # 3. Plot
    plt.figure(figsize=(10, 6))
    cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    cmap_bold = ListedColormap(['#0000FF', '#00FF00', '#FF0000'])

    # Filled contour
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)

    # Training points
    for c in unique_classes:
        mask = (y_train == c)
        plt.scatter(X_train[mask, 0], X_train[mask, 1], s=60, 
                    edgecolor='k', label=f"Class {c}")

    plt.title(f"KNN Decision Boundaries (k={k_value})")
    plt.show()