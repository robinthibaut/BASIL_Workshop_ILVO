import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # for neural networks
import tensorflow_probability as tfp  # for Bayesian neural networks

from sklearn.preprocessing import StandardScaler, MinMaxScaler  # for preprocessing
from sklearn.pipeline import Pipeline  # to create a pipeline

from os.path import join  # to join paths

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