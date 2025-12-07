import numpy as np
import math as m

def knn(data, u, k):
    def distance_euc(u, v):
        d = 0
        p = u.shape[0]
        for j in range(p):
            d += (u[j] - v[j]) ** 2
        return m.sqrt(d)

    def classe_major(liste_knn, classes):
        res = {c: len([e for e in liste_knn if e[1] == c]) for c in classes}
        return max(res, key=lambda i: res[i])

    X, y = data
    h, w = X.shape
    liste_knn = []
    classes = np.unique(y)

    for i in range(h):
        v = X[i, :]
        dist = distance_euc(u, v)
        liste_knn.append((dist, y[i]))

    liste_knn.sort(key=lambda tup: tup[0])
    c = classe_major(liste_knn[:k], classes)
    return c

def load_data_from_csv(filename="data.csv"):
    try:
        data = np.genfromtxt(filename, delimiter=",", dtype=str, skip_header=1)
        # Extract features (cols 0,1) and labels (col 2)
        X = data[:, :2].astype(float)
        y = data[:, 2]
        return X, y
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None