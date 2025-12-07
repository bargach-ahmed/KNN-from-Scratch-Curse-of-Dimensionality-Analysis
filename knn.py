import numpy as np
import math as m
import matplotlib.pyplot as plt


# ---------- KNN FUNCTION ----------
def knn(data, u, k):
    def distance_euc(u, v):
        d = 0 #distance
        p = u.shape[0]
        for j in range(p):
            d += (u[j] - v[j]) ** 2
        return m.sqrt(d)

    def classe_major(liste_knn, classes):
        #liste contient des tuples (distance, classe)
        #pour chaque classe on compte le nbr d'occurrences
        res = {c: len([e for e in liste_knn if e[1] == c]) for c in classes}
        return max(res, key=lambda i: res[i]) #max terme de value

    X, y = data
    h, w = X.shape
    liste_knn = []
    classes = np.unique(y) # les classes sans repetition

    for i in range(h):
        v = X[i, :] #la ligne i
        dist = distance_euc(u, v)
        liste_knn.append((dist, y[i]))

    liste_knn.sort(key=lambda tup: tup[0]) #tri selon dist
    c = classe_major(liste_knn[:k], classes)
    return c


# ---------- LOAD DATA FROM CSV ----------
file = "data.csv"  

data = np.genfromtxt(file, delimiter=",", dtype=str, skip_header=1)

# Extract features and labels
X_train = data[:, :2].astype(float)   # First 2 columns 
y_train = data[:, 2]                  # Last column = class labels


# ---------- QUERY POINT ----------
u_query = np.array([3, 5.0])   # Change this to test different points
k_value = 3


# ---------- RUN KNN ----------
prediction = knn((X_train, y_train), u_query, k_value)
print(f"Predicted Class: {prediction}")


# ---------- VISUALIZATION ----------
plt.figure(figsize=(8, 6))

classes = np.unique(y_train)
colors = {'A': 'blue', 'B': 'green', 'C': 'orange'}

# Plot each class separately
for c in classes:
    plt.scatter(X_train[y_train == c, 0],
                X_train[y_train == c, 1],
                s=80,
                color=colors[c],
                label=f"Class {c}")

# Plot query point
plt.scatter(u_query[0], u_query[1],
            color='red', marker='*', s=300,
            label=f"Query Point (Predicted: {prediction})")

plt.title(f"KNN Visualization (k={k_value})")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
