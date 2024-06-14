import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from src.benchmarks import NextHistogramDataset

print("\n PROPERTIES OF THE NEXT HISTOGRAM DATASET \n")

data_dir = 'Datasets/Data'
file_name = 'NextHistogramDataset_n_smpl50000__seq_len10__v_size15__seed42.pkl'

data_path = os.path.join(data_dir, file_name)

with open(data_path, "rb") as file: 
    dataset = pickle.load(file)


print("Statistical properties of the dataset:")
print(f"Mean of the tokens: {np.mean(dataset.X):.2f}, "
      f"STD of the tokens: {np.std(dataset.X):.2f}, "
      f"Median of the tokens: {np.median(dataset.X):.2f}")

print("\n")

distr = []

vocab_y = np.unique(dataset.y)  # Convert y to a set to get unique values, then convert back to list
total_sequences = dataset.y.shape[0] * dataset.y.shape[1]  # Assuming dataset.seq_len represents the length of each sequence

for i in vocab_y:
    h = len(dataset.y[np.where(dataset.y == i)])  # Count occurrences of each value in y
    h = (h / total_sequences) * 100  # Calculate the percentage
    distr.append(h)

plt.bar(vocab_y, distr, color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Percentage')
plt.title('Distribution of the labels')
plt.show()

vocab = np.arange(dataset.vocab_size)
distr = []
dataset.X = np.array(dataset.X)
for i in (vocab):
    z = len(dataset.X[np.where(dataset.X == i)])
    z = (z / ((len(dataset.X) * dataset.seq_len))) * 100
    distr.append(z)

plt.bar(vocab, distr, color='skyblue')
plt.xlabel('Tokens')
plt.ylabel('Percentage')
plt.title('Distribution of the tokens')
plt.show()

