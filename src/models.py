import tensorflow as tf
import tensorflow_probability as tfp


def get_normalizing_flow(ndim):
    # define distributions and bijectors and other objects that are needed
    tfd = tfp.distributions
    tfb = tfp.bijectors
    zdist = tfd.MultivariateNormalDiag(loc=[0.0] * ndim)
    num_layers = 2
    my_bijects = []

    # loop over desired bijectors and put into list
    for i in range(num_layers):
        # Syntax to make a MAF
        anet = tfb.AutoregressiveNetwork(
            params=2, hidden_units=[128, 128], activation="tanh"
        )
        ab = tfb.MaskedAutoregressiveFlow(anet)

        # Add bijector to list
        my_bijects.append(ab)

        # Now permute (!important!)
        permute = tfb.Permute(list(range(ndim))[::-1])
        my_bijects.append(permute)

    # put all bijectors into one "chain bijector"
    # that looks like one
    big_bijector = tfb.Chain(my_bijects)

    # make transformed dist
    td = tfd.TransformedDistribution(zdist, bijector=big_bijector)

    # declare the feature dimension
    x = tf.keras.Input(shape=(ndim,), dtype=tf.float32)

    # create a "placeholder" function that will be model output
    log_prob = td.log_prob(x)

    # use input (feature) and output (log prob) to make model
    model = tf.keras.Model(x, log_prob)
    initializer = tf.keras.initializers.GlorotNormal()
    for layer in model.layers:
        if hasattr(layer, "kernel_initializer"):
            layer.kernel_initializer = initializer

    return model, td


def neg_loglik(yhat, log_prob):
    """
    losses always take in label, prediction in keras. We do not have labels,
    but we still need to accept the arg to comply with Keras format.
    """

    return -log_prob
