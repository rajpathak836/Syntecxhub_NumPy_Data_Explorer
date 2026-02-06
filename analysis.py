
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Create screenshots directory safely
os.makedirs("screenshots", exist_ok=True)

data = np.load("data.npy")

print("Shape:", data.shape)
print("First 5 rows:\n", data[:5])

print("Mean:", np.mean(data, axis=0))
print("Median:", np.median(data, axis=0))
print("Std Dev:", np.std(data, axis=0))

# Reshaping & Broadcasting
reshaped = data.reshape(50, 10)
broadcasted = data + 10

# Performance comparison
start = time.time()
sum(data.tolist()[0])
python_time = time.time() - start

start = time.time()
np.sum(data[0])
numpy_time = time.time() - start

print("Python time:", python_time)
print("NumPy time:", numpy_time)

df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(1,6)])

sns.histplot(df["Feature_1"], kde=True)
plt.title("Distribution of Feature 1")
plt.savefig("screenshots/feature1_distribution.png")
plt.close()

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("screenshots/correlation_heatmap.png")
plt.close()

plt.plot(df.index, df["Feature_1"])
plt.title("Feature 1 Trend")
plt.savefig("screenshots/feature1_trend.png")
plt.close()

print("NumPy Data Explorer Completed Successfully")
