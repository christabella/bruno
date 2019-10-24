import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

import data_iter_conditional
import nn_extra_gauss
import nn_extra_nvp_conditional as nn_extra_nvp
from config_conditional import defaults

# GLOW
from glow import flow_layers as fl
from glow import nets

batch_size = 4
rng = np.random.RandomState(42)
rng_test = np.random.RandomState(317070)
seq_len = defaults.seq_len
n_context = defaults.n_context

nonlinearity = tf.nn.elu
weight_norm = True

train_data_iter = data_iter_conditional.ShapenetConditionalNPDataIterator(
    seq_len=seq_len,
    batch_size=batch_size,
    set='train',
    rng=rng,
    should_dequantise=False)
test_data_iter = data_iter_conditional.ShapenetConditionalNPDataIterator(
    seq_len=seq_len,
    batch_size=batch_size,
    set='test',
    rng=rng_test,
    should_dequantise=False)

obs_shape = train_data_iter.get_observation_size(
)  # (seq_len, 32, 32, channels=1)
label_shape = train_data_iter.get_label_size()  # (seq_len, 2)
print('obs shape', obs_shape)
print('label shape', label_shape)

ndim = np.prod(obs_shape[1:])
corr_init = np.ones((ndim, ), dtype='float32') * 0.1

optimizer = 'rmsprop'
learning_rate = 0.001
lr_decay = 0.999995
scale_gp_grad = 1.
gp_grad_schedule = {0: 0., 500: 0.5, 1000: 1.}
max_iter = 200000
save_every = 5000

glow_layers = []
nvp_dense_layers = []
gp_layer = None


def build_model(x, y_label, init=False, sampling_mode=False):
    """
    Args:
        x: shape=(config.batch_size,) + config.obs_shape)
           -> (batch_size, seq_len, 32, 32, channels=1)
        y: shape=(bs, seq_len, 2)

    This function is called with
    > model = tf.make_template('model', config.build_model)
    and creates all trainable variables

    If sampling_mode:
        return x_samples
    else:
        return log_probs, log_probs, log_probs

    """
    # Ensures that all nn_extra_nvp.*_wn layers have init=init
    with arg_scope([nn_extra_nvp.conv2d_wn, nn_extra_nvp.dense_wn], init=init):
        if len(glow_layers) == 0:
            build_glow_model()

        global gp_layer
        if gp_layer is None:
            gp_layer = nn_extra_gauss.GaussianRecurrentLayer(
                shape=(ndim, ), corr_init=corr_init)

        # (batch_size, seq_len, 32, 32, channels=1)
        x_shape = nn_extra_nvp.int_shape(x)
        x_bs = tf.reshape(
            x, (x_shape[0] * x_shape[1], x_shape[2], x_shape[3], x_shape[4]))
        x_bs_shape = nn_extra_nvp.int_shape(x_bs)

        y_label_shape = nn_extra_nvp.int_shape(y_label)
        y_label_bs = tf.reshape(
            y_label, (y_label_shape[0] * y_label_shape[1], y_label_shape[2]))
        y_label_bs = tf.layers.dense(
            y_label_bs,
            units=32,
            activation=tf.nn.leaky_relu,
            kernel_initializer=nn_extra_nvp.Orthogonal(),
            use_bias=True,
            name='labels_layer')

        log_det_jac = tf.zeros(x_bs_shape[0])
        # GLOW doesn't do any pretransformation from jittering
        # but maybe we might still need to do the scaling.
        y = x_bs

        # TODO: Replace RealNVP layers with GLOW layers.
        # construct forward pass through convolutional NVP layers.
        z = None
        for layer in glow_layers:
            y, log_det_jac, z = layer.forward_and_jacobian(y,
                                                           log_det_jac,
                                                           z,
                                                           y_label=y_label_bs)

        z = tf.concat([z, y], 3)
        # Followed by 6 256-unit dense layers of alternating partitions/masks.
        for layer in nvp_dense_layers:
            z, log_det_jac, _ = layer.forward_and_jacobian(z,
                                                           log_det_jac,
                                                           None,
                                                           y_label=y_label_bs)

        z_shape = nn_extra_nvp.int_shape(z)
        # Reshape z to (batch_size, seq_len, -1)
        # (last dimension is probably number of dimensions in the data, HxWxC)
        z_vec = tf.reshape(z, (x_shape[0], x_shape[1], -1))
        # The log det jacobian z_i/x_i for every i in sequence of length n.
        log_det_jac = tf.reshape(log_det_jac, (x_shape[0], x_shape[1]))

        log_probs = []
        z_samples = []

        with tf.variable_scope("one_step", reuse=tf.AUTO_REUSE) as scope:
            gp_layer.reset()
            if sampling_mode:
                if n_context > 0:
                    for i in range(n_context):
                        gp_layer.update_distribution(z_vec[:, i, :])
                    for i in range(seq_len):
                        z_sample = gp_layer.sample(nr_samples=1)
                        z_samples.append(z_sample)
                else:  # Sampling mode from just prior (no context)
                    for i in range(seq_len):
                        z_sample = gp_layer.sample(nr_samples=1)
                        z_samples.append(z_sample)
                        # Update each dimension of the latent space
                        gp_layer.update_distribution(z_vec[:, i, :])
            else:  # Training mode
                if n_context > 0:  # Some of sequence are context points
                    for i in range(n_context):
                        gp_layer.update_distribution(z_vec[:, i, :])

                    for i in range(n_context, seq_len):
                        latent_log_prob = gp_layer.get_log_likelihood(
                            z_vec[:, i, :])
                        log_prob = latent_log_prob + log_det_jac[:, i]
                        log_probs.append(log_prob)
                else:  # Sampling from prior
                    for i in range(seq_len):
                        latent_log_prob = gp_layer.get_log_likelihood(
                            z_vec[:, i, :])
                        log_prob = latent_log_prob + log_det_jac[:, i]
                        log_probs.append(log_prob)
                        gp_layer.update_distribution(z_vec[:, i, :])

        if sampling_mode:
            z_samples = tf.concat(z_samples, 1)
            z_samples_shape = nn_extra_nvp.int_shape(z_samples)
            z_samples = tf.reshape(z_samples,
                                   z_shape)  # (n_samples*seq_len, z_img_shape)

            for layer in reversed(nvp_dense_layers):
                z_samples, _ = layer.backward(z_samples,
                                              None,
                                              y_label=y_label_bs)

            x_samples = None
            for layer in reversed(nvp_layers):
                x_samples, z_samples = layer.backward(x_samples,
                                                      z_samples,
                                                      y_label=y_label_bs)

            # inverse logit
            x_samples = 1. / (1 + tf.exp(-x_samples))
            x_samples = tf.reshape(x_samples,
                                   (z_samples_shape[0], z_samples_shape[1],
                                    x_shape[2], x_shape[3], x_shape[4]))
            return x_samples

        # Reshape from (N, A, B, C) to (A, N, B, C)
        # Kind of "zipping" the log probs
        log_probs = tf.stack(log_probs, axis=1)

        return log_probs, log_probs, log_probs


def build_glow_model():

    # Appends to global nvp_layers.
    global glow_layers
    num_scales = 3
    num_filters = 64
    num_res_blocks = 6
    for scale in range(num_scales - 1):
        nvp_layers.append(
            # Checkerboard is binary mask for convolution
            nn_extra_nvp.CouplingLayerConv('checkerboard0',
                                           name='Checkerboard%d_1' % scale,
                                           nonlinearity=nonlinearity,
                                           weight_norm=weight_norm,
                                           num_filters=num_filters,
                                           num_res_blocks=num_res_blocks))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('checkerboard1',
                                           name='Checkerboard%d_2' % scale,
                                           nonlinearity=nonlinearity,
                                           weight_norm=weight_norm,
                                           num_filters=num_filters,
                                           num_res_blocks=num_res_blocks))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('checkerboard0',
                                           name='Checkerboard%d_3' % scale,
                                           nonlinearity=nonlinearity,
                                           weight_norm=weight_norm,
                                           num_filters=num_filters,
                                           num_res_blocks=num_res_blocks))
        nvp_layers.append(nn_extra_nvp.SqueezingLayer(name='Squeeze%d' %
                                                      scale))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('channel0',
                                           name='Channel%d_1' % scale,
                                           nonlinearity=nonlinearity,
                                           weight_norm=weight_norm,
                                           num_filters=num_filters,
                                           num_res_blocks=num_res_blocks))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('channel1',
                                           name='Channel%d_2' % scale,
                                           nonlinearity=nonlinearity,
                                           weight_norm=weight_norm,
                                           num_filters=num_filters,
                                           num_res_blocks=num_res_blocks))
        nvp_layers.append(
            nn_extra_nvp.CouplingLayerConv('channel0',
                                           name='Channel%d_3' % scale,
                                           nonlinearity=nonlinearity,
                                           weight_norm=weight_norm,
                                           num_filters=num_filters,
                                           num_res_blocks=num_res_blocks))
        nvp_layers.append(
            nn_extra_nvp.FactorOutLayer(scale, name='FactorOut%d' % scale))

    # final layer
    scale = num_scales - 1
    nvp_layers.append(
        nn_extra_nvp.CouplingLayerConv('checkerboard0',
                                       name='Checkerboard%d_1' % scale,
                                       nonlinearity=nonlinearity,
                                       weight_norm=weight_norm,
                                       num_filters=num_filters,
                                       num_res_blocks=num_res_blocks))
    nvp_layers.append(
        nn_extra_nvp.CouplingLayerConv('checkerboard1',
                                       name='Checkerboard%d_2' % scale,
                                       nonlinearity=nonlinearity,
                                       weight_norm=weight_norm,
                                       num_filters=num_filters,
                                       num_res_blocks=num_res_blocks))
    nvp_layers.append(
        nn_extra_nvp.CouplingLayerConv('checkerboard0',
                                       name='Checkerboard%d_3' % scale,
                                       nonlinearity=nonlinearity,
                                       weight_norm=weight_norm,
                                       num_filters=num_filters,
                                       num_res_blocks=num_res_blocks))
    nvp_layers.append(
        nn_extra_nvp.CouplingLayerConv('checkerboard1',
                                       name='Checkerboard%d_4' % scale,
                                       nonlinearity=nonlinearity,
                                       weight_norm=weight_norm,
                                       num_filters=num_filters,
                                       num_res_blocks=num_res_blocks))
    nvp_layers.append(
        nn_extra_nvp.FactorOutLayer(scale, name='FactorOut%d' % scale))


def build_nvp_dense_model():
    """No convolutional residual layers."""
    global nvp_dense_layers

    for i in range(6):  # 6 dense layers
        mask = 'even' if i % 2 == 0 else 'odd'
        name = '%s_%s' % (mask, i)
        nvp_dense_layers.append(
            nn_extra_nvp.CouplingLayerDense(mask,
                                            name=name,
                                            nonlinearity=nonlinearity,
                                            n_units=256,
                                            weight_norm=weight_norm))


def loss(log_probs):
    return -tf.reduce_mean(log_probs)
