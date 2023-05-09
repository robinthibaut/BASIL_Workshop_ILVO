import tensorflow as tf  # for neural networks
import numpy as np  # for numerical operations
import tensorflow_probability as tfp  # for Bayesian neural networks
from tensorflow_probability import distributions as tfd  # for distributions


def neg_log_likelihood(x, rv_x):
    """Negative log likelihood of the data under the distribution."""
    return -rv_x.log_prob(x)


# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                2 * n,
                dtype=dtype,
                initializer=lambda shape, dtype: random_gaussian_initializer(
                    shape, dtype
                ),
                trainable=True,
            ),
            # tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(
                        loc=t[..., :n],
                        scale=1e-5 + 1e-2 * tf.nn.softplus(c + t[..., n:]),
                        # softplus ensures positivity and avoids numerical instability
                    ),
                    reinterpreted_batch_ndims=1,  # each weight is independent
                )  # reinterpreted_batch_ndims=1 means that the last dimension is the event dimension
            ),
        ]
    )


# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1
                )
            ),
        ]
    )


def random_gaussian_initializer(shape, dtype="float32"):
    n = int(shape / 2)
    loc_norm = tf.random_normal_initializer(mean=0.0, stddev=0.1)
    loc = tf.Variable(initial_value=loc_norm(shape=(n,), dtype=dtype))
    scale_norm = tf.random_normal_initializer(mean=-3.0, stddev=0.1)
    scale = tf.Variable(initial_value=scale_norm(shape=(n,), dtype=dtype))
    return tf.concat([loc, scale], 0)


def probabilistic_variational_model(
        input_shape: tuple,
        output_shape: tuple,
        learn_r: float = 0.001,
        num_components: int = 1,
):
    """
    Probabilistic variational model for regression.
    :param input_shape: tuple, shape of the input data
    :param output_shape: tuple, shape of the output data
    :param learn_r: float, learning rate
    :param num_components: int, number of components in the mixture model
    :return: tf.keras.Sequential, probabilistic variational model
    """
    params_size = tfp.layers.MixtureNormal.params_size(num_components, output_shape[-1])  # Number of parameters
    kl_weight = 1 / input_shape[0]  # Weight for the KL divergence
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_shape[1], input_shape[2])),  # Input layer
            tf.keras.layers.Conv1D(
                filters=4,
                kernel_size=2,
                padding="same",
                kernel_initializer=tf.keras.initializers.Zeros(),
            ),
            tf.keras.layers.MaxPool1D(pool_size=2),  # Pooling layer
            tf.keras.layers.Flatten(),
            tfp.layers.DenseVariational(
                units=12,
                make_prior_fn=prior_trainable,
                make_posterior_fn=posterior_mean_field,
                kl_weight=kl_weight,
                kl_use_exact=True,
                name="var1",
                activation="relu",
            ),  # Hidden layer 1
            tf.keras.layers.Dense(params_size),  # Hidden layer 2
            tfp.layers.MixtureNormal(num_components, output_shape[-1]),  # Mixture layer
        ],
        name="model",
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_r)  # Optimizer
    model.compile(optimizer=optimizer, loss=neg_log_likelihood)  # Compile model with loss and optimizer

    return model


@tf.function
def compute_kl_divs(y_true, y_pred):
    ideal_dist = tfp.distributions.Normal(loc=y_true, scale=0.1)
    predicted_dist = tfp.distributions.Normal(loc=y_pred.mean(), scale=y_pred.stddev())
    return tfp.distributions.kl_divergence(ideal_dist, predicted_dist)
