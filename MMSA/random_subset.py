# this script/file contains randomly generated indices for different percentages of MOSEI
import random
import numpy as np
N = 16326 # mosei
p=.9

# Calculate the number of indices to sample based on the percentage
num_samples = int(p * N)

# Generate a list of random indices up to N
all_indices = list(range(N))

# Sample random indices without replacement
subset_indices = np.sort(random.sample(all_indices, num_samples))

# Save the indices to a file
filename = f'subset_{p}_indices.txt'  # Replace with your desired filename

with open(filename, 'w') as file:
    for index in subset_indices:
        file.write(str(index) + '\n')

print(f"Subset indices saved to {filename}")