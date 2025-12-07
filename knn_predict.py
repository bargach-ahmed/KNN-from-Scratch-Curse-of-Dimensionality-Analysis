import numpy as np
from knn_core import knn, load_data_from_csv

print("--- KNN Prediction ---")

# 1. Load Data
X_train, y_train = load_data_from_csv()

if X_train is not None:
    try:
        # 2. Get User Input
        # Note: Enter numbers separated by SPACE (e.g., 3 3)
        user_input = input("Enter X Y coordinate (e.g., 5.1 3.5): ").split()
        x1, x2 = map(float, user_input)
        
        k_value = int(input("Enter K value (e.g., 3): "))
        
        # 3. Predict
        u_query = np.array([x1, x2])
        prediction = knn((X_train, y_train), u_query, k_value)
        
        print(f"\nPredicted Class: {prediction}")
        
    except ValueError:
        print("Error: Please enter valid numbers separated by a space.")