from typing import List, Callable, Dict, Any, Tuple, NamedTuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from tensorflow.python.ops import template as template_ops
from tqdm import tqdm

from glow import flow_layers as fl
from glow import tf_ops
from glow import tf_ops as ops

K = tf.keras.backend
keras = tf.keras


class OpenAITemplate(NamedTuple):
    """
    A shallow neural network used by GLOW paper:
    * https://github.com/openai/glow

    activation_fn: activation function used after each conv layer
    width: number of filters in the shallow network
    """
    activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu
    width: int = 32

    def create_template_fn(
            self,
            name: str,
    ) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        """
        Creates simple shallow network. Note that this function will return a
        tensorflow template.
        Args:
            name: a scope name of the network
        Returns:
            a template function
        """
        def _shift_and_log_scale_fn(x: tf.Tensor, y_label: tf.Tensor = None):
            """NN is a shallow, 3 convolutions with 512 units: 3x3, 1x1, 3x3, the last one returns shift and logscale
            """
            shape = K.int_shape(x)
            num_channels = shape[3]

            with tf.variable_scope("BlockNN"):
                h = x
                # Concatenate conditioning labels with x.
                # Just in the shift and log scale fn should be fine...
                h = ops.conv2d("l_1", h, self.width)
                depth = K.int_shape(h)[-1]
                label_size = K.int_shape(y_label)[-1]
                dense_w = tf.get_variable(
                    "dense_w",
                    shape=(label_size, depth),
                    initializer=tf.contrib.layers.xavier_initializer())
                dense_b = tf.get_variable(
                    "dense_b",
                    shape=(depth, ),
                    initializer=tf.contrib.layers.xavier_initializer())

                conditioning_y = tf.nn.xw_plus_b(y_label, dense_w, dense_b)
                h = h + conditioning_y[:, None, None, :]
                h = self.activation_fn(h)  # 3x3 filter
                h = self.activation_fn(
                    ops.conv2d("l_2", h, self.width, filter_size=[1, 1]))
                # create shift and log_scale with zero initialization
                shift_log_scale = ops.conv2d_zeros("l_last", h,
                                                   2 * num_channels)
                shift = shift_log_scale[:, :, :, 0::2]
                log_scale = shift_log_scale[:, :, :, 1::2]
                log_scale = tf.clip_by_value(log_scale, -15.0, 15.0)
                return shift, log_scale

        return template_ops.make_template(name, _shift_and_log_scale_fn)


def step_flow(name: str,
              shift_and_log_scale_fn: Callable[[tf.Tensor], tf.Tensor]
              ) -> Tuple[fl.ChainLayer, fl.ActnormLayer]:
    """Create single step of the Glow model:

        1. actnorm
        2. invertible conv
        3. affine coupling layer

    Returns:
        step_layer: a flow layer which perform 1-3 operations
        actnorm: a reference of actnorm layer from step 1. This reference can be
            used to initialize this layer using data dependent initialization
    """
    actnorm = fl.ActnormLayer()
    layers = [
        actnorm,
        fl.InvertibleConv1x1Layer(),
        fl.AffineCouplingLayer(shift_and_log_scale_fn=shift_and_log_scale_fn),
    ]
    return fl.ChainLayer(layers, name=name), actnorm


def initialize_actnorms(
        sess: tf.Session(),
        feed_dict_fn: Callable[[], Dict[tf.Tensor, np.ndarray]],
        actnorm_layers: List[fl.ActnormLayer],
        num_steps: int = 100,
        num_init_iterations: int = 10,
) -> None:
    """Initialize actnorm layers using data dependent initialization

    Args:
        sess: an instance of tf.Session
        feed_dict_fn: a feed dict function which return feed_dict to the tensorflow
            sess.run function.
        actnorm_layers: a list of actnorms to initialize
        num_steps: number of batches to used for iterative initialization.
        num_init_iterations: a get_ddi_init_ops parameter. For more details
            see the implementation.
    """
    for actnorm_layer in tqdm(actnorm_layers):
        init_op = actnorm_layer.get_ddi_init_ops(num_init_iterations)
        for i in range(num_steps):
            sess.run(init_op, feed_dict=feed_dict_fn())


def create_simple_flow(num_steps: int = 1,
                       num_scales: int = 3,
                       num_bits: int = 5,
                       template_fn: Any = OpenAITemplate()
                       ) -> Tuple[List[fl.FlowLayer], List[fl.ActnormLayer]]:
    """Create Glow model. This implementation may slightly differ from the
    official one. For example the last layer here is the fl.FactorOutLayer

    Args:
        num_steps: number of steps per single scale; K parameter from the paper
        num_scales: number of scales, a L parameter from the paper. Each scale
            reduces the tensor spatial dimension by 2.
        num_bits: input image quantization
        template_fn: a template function used in AffineCoupling layer

    Returns:
        layers: a list of layers which define normalizing flow
        actnorms: a list of actnorm layers which can be initialized using data
            dependent initialization. See: initialize_actnorms() function.
    """
    layers = [fl.QuantizeImage(num_bits=num_bits)]
    actnorm_layers = []
    for scale in range(num_scales):
        scale_name = f"Scale{scale+1}"
        scale_steps = []
        for s in range(num_steps):
            name = f"Step{s+1}"
            step_layer, actnorm_layer = step_flow(
                name=name,
                shift_and_log_scale_fn=template_fn.create_template_fn(name))
            scale_steps.append(step_layer)
            actnorm_layers.append(actnorm_layer)

        layers += [
            fl.SqueezingLayer(name=scale_name),
            fl.ChainLayer(scale_steps, name=scale_name),
            fl.FactorOutLayer(name=scale_name),
        ]

    return layers, actnorm_layers
