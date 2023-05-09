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

# Let's demonstrate the approach on the second pair.
pn = 'pair2'  # pair name
pair2 = join(data_dir, pn)  # path to the second pair
X_train = np.load(join(pair2, 'X_train.npy'))  # load the predictors for the training set
y_train = np.load(join(pair2, 'y_train.npy'))  # load the target for the training set
X_test = np.load(join(pair2, 'X_test.npy'))  # load the predictors for the test set
y_test = np.load(join(pair2, 'y_test.npy'))  # load the target for the test set

# X contains the breakthrough curves at the two sensors.
# It has a total of 101 time steps, and two predictors (one for each sensor).

# y is the arrival time of the contaminant at the river, in days.

# Let's have a look at the data.
print(X_train.shape)  # (7999, 101, 2) (I intended to have 8000 samples, but I accidentally deleted one :))
print(y_train.shape)  # (7999, 1)
print(X_test.shape)  # (2000, 101, 2)
print(y_test.shape)  # (2000, 1)

# Let's plot the predictors for some samples in the training set.
n_to_show = 1000  # number of samples to show
# Let's first load the time steps.
t = np.load(join(data_dir, 'times.npy')).reshape(-1, )  # load the time steps and flatten the array

# Let's plot the predictors for the first n_to_show samples in the training set.
# The first predictor is the concentration at the first sensor.
plt.plot(np.repeat(t, n_to_show).reshape(-1, n_to_show), X_train[:n_to_show, :, 0].T, color='blue')
plt.xlabel('Time (days)')
plt.ylabel('Concentration (mg/L)')
plt.title('Predictor 1')
plt.show()

# The second predictor is the concentration at the second sensor.
plt.plot(np.repeat(t, n_to_show).reshape(-1, n_to_show), X_train[:n_to_show, :, 1].T, color='red')
plt.xlabel('Time (days)')
plt.ylabel('Concentration (mg/L)')
plt.title('Predictor 2')
plt.show()

# The target is the arrival time of the contaminant at the river, in days.
# It is univariate, so we can directly plot the distribution of the target.
plt.hist(y_train)
plt.xlabel('Arrival time (days)')
plt.ylabel('Frequency')
plt.title('Distribution of the target')
plt.show()

# Let's now create a pipeline to preprocess the data.
# In machine learning, it is often a good idea to scale the data before applying any algorithm.
# Here, we will use the StandardScaler from scikit-learn.

# You might have observed that the predictors sometimes contain negative values or very small values.
# This is due to numerical effect, and we will not go into the details here.
# Since the concentration cannot be negative, we will first replace the negative values by zero.
X_train[X_train < 1e-5] = 0
X_test[X_test < 1e-5] = 0

# Let's first create a preprocessor for the predictors.
pipeline_x = Pipeline([('scaler', StandardScaler()),
                       ('minmax', MinMaxScaler(feature_range=(0, 1)))])
# Let's now fit the scaler on the training data.
pipeline_x.fit(X_train.reshape(-1, 2))
# Let's now transform the training data.
X_train_scaled = pipeline_x.transform(X_train.reshape(-1, 2)).reshape(-1, 101, 2)
# Let's now transform the test data.
X_test_scaled = pipeline_x.transform(X_test.reshape(-1, 2)).reshape(-1, 101, 2)

# Let's now create a pipeline for the target
pipeline_y = Pipeline([('scaler', StandardScaler()),
                       ('minmax', MinMaxScaler(feature_range=(0, 1)))])
# Let's now fit the scaler on the training data.
pipeline_y.fit(y_train)
# Let's now transform the training data.
y_train_scaled = pipeline_y.transform(y_train)
# Let's now transform the test data.
y_test_scaled = pipeline_y.transform(y_test)

# I already created a neural network architecture for you.
# you can just import it as follows.
from basil.functions import probabilistic_variational_model

# Let's now create a model.
# You just need to specify the input and output shapes.
model = probabilistic_variational_model(input_shape=X_train_scaled.shape,
                                        output_shape=y_train_scaled.shape,
                                        learn_r=0.001,)

# Let's now train the model.
# You can specify the number of epochs and the batch size.
# define an early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,  # number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # restore the best model
)

# fit the model
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=500,  # number of epochs - one epoch is one iteration over the entire training set
    batch_size=32,  # batch size - number of samples per gradient update
    verbose=1,  # verbose mode - 0: silent, 1: not silent
    validation_split=0.1,  # validation split - 10% of the training data will be used for validation
    callbacks=[early_stopping],  # early stopping  - stop training when the validation loss is not decreasing anymore
)

# Let's now plot the training history.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training history')
plt.legend(['Training loss', 'Validation loss'])
plt.show()

# Great! The training loss and validation loss are decreasing.
# Let's now evaluate the model on the test set.
# Let's first predict the target for one example in the test set.
y_post1 = model(X_test_scaled[0].reshape(1, -1, 2))

# y_post1 is the posterior distribution of the target conditioned on the predictor X_test_scaled[0].
# Let's sample from this distribution.
y_post1_sample = y_post1.sample(1000).numpy().reshape(-1, )  # sample from the posterior distribution
# Let's now plot the distribution of the target.
# It is always a good idea to plot the posterior distribution of the target on top of the prior distribution.
plt.hist(y_train_scaled, density=True, label='Prior')
plt.hist(y_post1_sample, density=True, alpha=0.8, label='Posterior')
# plot the true value
plt.axvline(y_test_scaled[0], color='red', label='True value')
plt.xlim([0, 1])
plt.xlabel('Arrival time (days - scaled)')
plt.ylabel('Frequency')
plt.title('Distribution of the target')
plt.legend()
plt.show()


# What do you think of the result?

# Let's compute our metric of interest.
def kl_div(y_true, y_pred):
    ideal_dist = tfp.distributions.Normal(loc=y_true, scale=0.1)
    predicted_dist = tfp.distributions.Normal(loc=y_pred.mean(), scale=y_pred.std())
    return tfp.distributions.kl_divergence(ideal_dist, predicted_dist)


# Let's first compute the KL divergence between the prior and the posterior.
kl_div1 = kl_div(y_test_scaled[0], y_post1_sample).numpy()

# What's the score?

# Now let's predict the whole test set
y_post = model(X_test_scaled)

# I already created a function to compute the KL divergence on the whole test set.
from basil.functions import compute_kl_divs

kl_divs = compute_kl_divs(y_test_scaled, y_post).numpy()
# save the results for later
np.save(join(output_dir, f'kl_div_{pn}.npy'), kl_divs)

# Let's now plot the distribution of the KL divergence.
plt.hist(kl_divs, density=True)
plt.ylim([0, 1])
plt.xlabel('KL divergence')
plt.ylabel('Frequency')
plt.title('Distribution of the KL divergence')
plt.savefig(join(output_dir, f'kl_div_{pn}.png'), dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Now repeat the same experiment with the other pair of predictors.
