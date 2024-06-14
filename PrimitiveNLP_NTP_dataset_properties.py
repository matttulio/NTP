import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from src.benchmarks import PrimitiveNLP_NTP
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency

print("\n")
print("PROPERTIES OF THE PRIMITIVE NLP NTP DATASET")
print("\n")

data_dir = 'Datasets/Data'
file_name = 'primitive_NLP_NTP_dataset_n_smpl50000__seq_len10__cont_win3__'\
        'v_size78__emb_dim50__emb_typeglove.6B.50d__seed42__d_par2.pkl'

data_path = os.path.join(data_dir, file_name)

with open(data_path, "rb") as file: 
    dataset = pickle.load(file)


unique_rows, counts = np.unique(dataset.X, axis=0, return_counts=True)
unique_row_counts = dict(zip(map(tuple, unique_rows), counts))

print("Percentage of unique samples: ", (len(unique_row_counts) / dataset.num_samples) * 100, "\n")

print("Statistical properties of the dataset:")
print(f"Mean of the tokens: {np.mean(dataset.X):.2f}, "
      f"STD of the tokens: {np.std(dataset.X):.2f}, "
      f"Median of the tokens: {np.median(dataset.X):.2f}")
print("\n")

distr = []

# Count how many times a token appears
for i in (dataset.vocab):
    z = len(dataset.X[np.where(dataset.X == i)])
    distr.append(z)

# Compute the frequencies of each token
num_tok = np.count_nonzero(dataset.X)
distr = np.array(distr) / num_tok

# Sort the frequencies in descending order
distr[::-1].sort()
distr = distr.tolist()

plt.plot(range(1, dataset.vocab_size + 1), distr, '-', color = 'blue', label = 'Observed distribution')

x_values = np.linspace(1, dataset.vocab_size, dataset.vocab_size)  # Generating x values for the function
y_values = np.max(distr) / (x_values + 0) ** (1)


plt.plot(x_values, y_values, color='skyblue', label = 'Zipf`s Law: k = 1')
plt.xlabel('Degree')
plt.ylabel('Frequency')
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.title('Distribution of the tokens')
plt.show()


# Define the function to fit
def func(x, k, b):
    return  1 / (x + b) ** k

# Plot the data
plt.plot(x_values, distr, '-', color='blue', label = 'Observed distribution' )

# Fit the function to the data
popt, pcov = curve_fit(func, x_values, distr, (1, 2.7))

# Plot the fitted function
plt.plot(x_values, func(x_values, *popt), color='skyblue', label=f'Zipf`s Law:  k={popt[0]:.2f}, {popt[1]:.2f}')

# Plot settings
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Distribution of the tokens')
plt.legend()
plt.show()

print("Fit the distribution parameters")
for i, param in enumerate(popt):
    print(f"Optimal parameter {i+1}: {param:.2f}")
print()

# Compute the fitted values
fitted_values = func(x_values, *popt)
table = np.vstack((distr, fitted_values))

# Calculate the statistic of the plot to determine wether the distribution is Zipfian
res = chi2_contingency(table, correction=False)

print(f"Chi square test: {res.statistic:.3f}, {res.pvalue:.3f}")
print("\n")