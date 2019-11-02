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

glow_model = None
gp_layer = None


def build_model(x, y_label, init=False, sampling_mode=False):
    """
    Args:
        x: float32 placeholder with shape=(config.batch_size,) + config.obs_shape)
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
        if glow_model is None:
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
        # Try commenting this out, what's the point of it?
        # y_label_bs = tf.layers.dense(
        #     y_label_bs,
        #     units=32,
        #     activation=tf.nn.leaky_relu,
        #     kernel_initializer=nn_extra_nvp.Orthogonal(),
        #     use_bias=True,
        #     name='labels_layer')

        log_det_jac = tf.zeros(x_bs_shape[0])
        # GLOW doesn't do any pretransformation from jittering
        # but maybe we might still need to do the scaling.
        x_bs, log_det_jac = nn_extra_nvp.dequantization_forward_and_jacobian(
            x_bs, log_det_jac)

        # TODO: Replace RealNVP layers with GLOW layers.
        # construct forward pass through convolutional NVP layers.
        z = None
        # This is not having a sequence length... wait do we individually pass each image through its own normalizing flow?
        # Now we have batch size * seq len images in x_bs
        input_flow = fl.InputLayer(x_bs, y_label_bs)
        # forward flow
        output_flow = glow_model(input_flow, forward=True)
        # backward flow
        # reconstruction = glow_model(output_flow, forward=False)
        # flow is a tuple of three tensors
        x, log_det_jac, z, y_label = output_flow
        #  x=[64, 2, 2, 16]	z=[64, 2, 2, 240]	logdet=[64]
        # z is None...
        z = tf.concat([z, x], 3)  # Join the split channels back
        # [64, 2, 2, 256]
        z_shape = nn_extra_nvp.int_shape(z)
        # Reshape z to (batch_size, seq_len, -1)
        # (last dimension is probably number of dimensions in the data, HxWxC)
        z_vec = tf.reshape(z, (x_shape[0], x_shape[1], -1))  # 4, 16, 1024
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
                        # 64, 1, 480
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

            x_samples = None
            # TODO do backward flow on glow model
            # for layer in reversed(glow_layers):
            #     x_samples, z_samples = layer.backward(x_samples,
            #                                           z_samples,
            #                                           y_label=y_label_bs)

            # inverse logit
            # x_samples = 1. / (1 + tf.exp(-x_samples))
            x_samples = tf.reshape(x_samples,
                                   (z_samples_shape[0], z_samples_shape[1],
                                    x_shape[2], x_shape[3], x_shape[4]))
            return x_samples

        # Reshape from (N, A, B, C) to (A, N, B, C)
        # Kind of "zipping" the log probs
        log_probs = tf.stack(log_probs, axis=1)
        # log probs: (4, 15)
        return log_probs, log_probs, log_probs


def build_glow_model():
    # Appends to global nvp_layers.
    global glow_model
    layers, actnorm_layers = nets.create_simple_flow(
        num_steps=32,  # same as K
        num_scales=4,  # same as L parameter
        # template_fn=nets.OpenAITemplate(width=512))
        template_fn=nets.OpenAITemplate(width=8))
    model = fl.ChainLayer(layers)
    # x_bs = tf.placeholder(tf.float32, [16, 64, 64, 3])
    # flow = fl.InputLayer(x_bs)
    glow_model = model


def loss(log_probs):
    return -tf.reduce_mean(log_probs)
