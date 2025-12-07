import numpy as np
import matplotlib.pyplot as plt

def generate_distances(n_samples, n_dim):
    # 1. Generate random data uniformly between 0 and 1
    # Shape: (1000 points, n_dim features)
    X = np.random.rand(n_samples, n_dim)
    
    # 2. Pick a reference point (e.g., the first point)
    u = X[0]
    
    # 3. Calculate Euclidean distance from u to all other points
    # We use vectorization: sqrt(sum((x - u)^2))
    diff = X - u
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # Remove the distance to itself (which is 0)
    return distances[1:]

# --- Settings ---
N = 1000 # Number of points

# --- Run for Low Dimension (2D) ---
dist_low = generate_distances(N, n_dim=2)

# --- Run for High Dimension (3000D) ---
dist_high = generate_distances(N, n_dim=3000)

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Low Dimension
axes[0].hist(dist_low, bins=30, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title(f"2 Dimensions\n(Min={dist_low.min():.2f}, Max={dist_low.max():.2f})")
axes[0].set_xlabel("Euclidean Distance")
axes[0].set_ylabel("Frequency")

# Plot 2: High Dimension
axes[1].hist(dist_high, bins=30, color='red', alpha=0.7, edgecolor='black')
axes[1].set_title(f"3000 Dimensions\n(Min={dist_high.min():.2f}, Max={dist_high.max():.2f})")
axes[1].set_xlabel("Euclidean Distance")
axes[1].set_ylabel("Frequency")

# Calculate "Relative Contrast"
contrast_low = (dist_low.max() - dist_low.min()) / dist_low.min()
contrast_high = (dist_high.max() - dist_high.min()) / dist_high.min()

print(f"2D Relative Contrast:   {contrast_low:.4f} (Distances are very different)")
print(f"3000D Relative Contrast: {contrast_high:.4f} (Distances are almost the same)")

plt.show()