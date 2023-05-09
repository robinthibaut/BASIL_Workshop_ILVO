import tensorflow as tf  # for neural networks
import numpy as np
import tensorflow_probability as tfp  # for Bayesian neural networks
from tensorflow_probability import distributions as tfd


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
                        scale=1e-5 + 1e-2 * tf.nn.softplus(c + t[..., n:]),  # softplus ensures positivity and avoids numerical instability
                    ),
                    reinterpreted_batch_ndims=1, # each weight is independent
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