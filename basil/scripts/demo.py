from os.path import join  # to join paths

import matplotlib.pyplot as plt
import numpy as np

from basil.config import Directories  # this is the basil directory

main_dir = Directories.main_dir  # this is the basil directory
data_dir = Directories.data_dir  # this is the data directory
output_dir = Directories.results_dir  # this is the output directory

# Let's demonstrate the approach on the second pair.
pn = "pair2"  # pair name
pair2 = join(data_dir, pn)  # path to the second pair
X_train = np.load(
    join(pair2, "X_train.npy")
)  # load the predictors for the training set
y_train = np.load(join(pair2, "Y_train.npy"))  # load the target for the training set
X_test = np.load(join(pair2, "X_test.npy"))  # load the predictors for the test set
y_test = np.load(join(pair2, "Y_test.npy"))  # load the target for the test set

# X contains the breakthrough curves at the two sensors.
# It has a total of 101 time steps, and two predictors (one for each sensor).

# y is the arrival time of the contaminant at the river, in days.

# Let's have a look at the data.
print(
    X_train.shape
)  # (7999, 101, 2) (I intended to have 8000 samples, but I accidentally deleted one :))
print(y_train.shape)  # (7999, 1)
print(X_test.shape)  # (2000, 101, 2)
print(y_test.shape)  # (2000, 1)

# Let's plot the predictors for some samples in the training set.
n_to_show = 1000  # number of samples to show
# Let's first load the time steps.
t = np.load(join(data_dir, "times.npy")).reshape(
    -1,
)  # load the time steps and flatten the array

# Let's plot the predictors for the first n_to_show samples in the training set.
# The first predictor is the concentration at the first sensor.
plt.plot(
    np.repeat(t, n_to_show).reshape(-1, n_to_show),
    X_train[:n_to_show, :, 0].T,
    color="blue",
)
plt.xlabel("Time (days)")
plt.ylabel("Concentration (mg/L)")
plt.title("Predictor 1")
plt.show()

# The second predictor is the concentration at the second sensor.
plt.plot(
    np.repeat(t, n_to_show).reshape(-1, n_to_show),
    X_train[:n_to_show, :, 1].T,
    color="red",
)
plt.xlabel("Time (days)")
plt.ylabel("Concentration (mg/L)")
plt.title("Predictor 2")
plt.show()

# The target is the arrival time of the contaminant at the river, in days.
# It is univariate, so we can directly plot the distribution of the target.
plt.hist(y_train)
plt.xlabel("Arrival time (days)")
plt.ylabel("Frequency")
plt.title("Distribution of the target")
plt.show()
