from os.path import join  # to join paths

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp  # for Bayesian neural networks
from sklearn.preprocessing import PowerTransformer  # for preprocessing

from basil.config import Directories  # this is the basil directory

main_dir = Directories.main_dir  # this is the basil directory
data_dir = Directories.data_dir  # this is the data directory
output_dir = Directories.results_dir  # this is the output directory

# Let's load the three arrays of KL divergences.
kl1 = np.load(join(output_dir, 'kl_div_pair1.npy'))
kl2 = np.load(join(output_dir, 'kl_div_pair2.npy'))
kl3 = np.load(join(output_dir, 'kl_div_pair3.npy'))

# Let's plot the KL divergences for the three pairs.
plt.hist(kl1, density=True, alpha=0.5, label='Pair 1')
plt.hist(kl2, density=True, alpha=0.5, label='Pair 2')
plt.hist(kl3, density=True, alpha=0.5, label='Pair 3')
plt.xlabel('KL divergence')
plt.ylabel('Frequency')
plt.legend()
plt.xlim([0, 20])
plt.show()

# The best pair is quite clear.

# For more complex cases, it may not be so clear.

# In that case, you can use the following function to find the best pair.

names = ['pair1', 'pair2', 'pair3']  # names of the pairs
kl_divs = [kl1, kl2, kl3]  # KL divergences of the pairs
min_kl = np.inf
scores_kl = np.zeros(len(kl_divs))

for i in range(len(names)):
    kli = kl_divs[i]
    # remove nan and inf values from the array
    kli = kli[np.isfinite(kli)]
    # remove values > 10
    kli = kli[kli < 10]

    # apply a box-cox transformation to the data
    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    kli = transformer.fit_transform(kli.reshape(-1, 1)).flatten()

    # compute the mean of the transformed data
    mean_kl = np.mean(kli)
    # compute the standard deviation of the transformed data
    std_kl = np.std(kli)

    # define a normal distribution with the mean and std computed above
    normal = tfp.distributions.Normal(loc=mean_kl, scale=std_kl)
    # define an ideal normal distribution with mean 0 and std 0.1
    ideal = tfp.distributions.Normal(loc=0, scale=0.1)

    # compute the kl divergence between the two distributions
    kl = tfp.distributions.kl_divergence(ideal, normal)

    scores_kl[i] = kl

print(scores_kl)