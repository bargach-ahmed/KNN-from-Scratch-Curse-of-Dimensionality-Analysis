# FILE: knn_plot.py
import numpy as np
import matplotlib.pyplot as plt
from knn_core import knn, load_data_from_csv

X_train, y_train = load_data_from_csv()

if X_train is not None:
    # Example Query Point
    u_query = np.array([5.1, 3.8])
    k_value = 3
    prediction = knn((X_train, y_train), u_query, k_value)

    plt.figure(figsize=(8, 6))
    
    # Plot Training Data
    classes = np.unique(y_train)
    colors = {'A': 'blue', 'B': 'green', 'C': 'orange'}
    
    for c in classes:
        mask = (y_train == c)
        plt.scatter(X_train[mask, 0], X_train[mask, 1], 
                    s=80, color=colors.get(c, 'gray'), label=f"Class {c}")

    # Plot Query Point
    plt.scatter(u_query[0], u_query[1], color='red', marker='*', s=300, 
                label=f"Query: {prediction}")

    plt.title(f"KNN Visualization (k={k_value})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()