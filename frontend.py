import torch
import torch.nn as nn
import math


class SquaredModulus(nn.Module):
    def __init__(self):
        super(SquaredModulus, self).__init__()
        self._pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        output = 2 * self._pool(x ** 2)
        return output.permute(0, 2, 1)


class GaborConv1D(nn.Module):
    def __init__(self):
        super(GaborConv1D, self).__init__()


class GaborConstraint(nn.Module):
  # Constraint mu and sigma, in radians.

    def __init__(self, kernel_size):
        super(GaborConstraint, self).__init__()
        self._kernel_size = kernel_size

    def forward(self, x):
        mu_lower = 0.
        mu_upper = math.pi
        sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
        sigma_upper = self._kernel_size * math.sqrt(2 * math.log(2)) / math.pi
        clipped_mu = tf.clip_by_value(kernel[:, 0], mu_lower, mu_upper)
        clipped_sigma = tf.clip_by_value(kernel[:, 1], sigma_lower, sigma_upper)
        return tf.stack([clipped_mu, clipped_sigma], axis=1)


class GaborConv1D(nn.Module):
    """Implements a convolution with filters defined as complex Gabor wavelets.

    These filters are parametrized only by their center frequency and
    the full-width at half maximum of their frequency response.
    Thus, for n filters, there are 2*n parameters to learn.
    """

    def __init__(self, filters, kernel_size, strides, padding, use_bias,
               input_shape, kernel_initializer, kernel_regularizer, name,
               trainable, sort_filters=False):
        super().__init__()
        self._filters = filters // 2
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._sort_filters = sort_filters
        # Weights are the concatenation of center freqs and inverse bandwidths.
        self._kernel = self.add_weight(
            name='kernel',
            shape=(self._filters, 2),
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
            trainable=trainable,
            constraint=GaborConstraint(self._kernel_size))
        if self._use_bias:
            self._bias = self.add_weight(name='bias', shape=(self._filters * 2,))

    def forward(self, x):
        kernel = self._kernel.constraint(self._kernel)
        if self._sort_filters:
            filter_order = torch.argsort(kernel[:, 0])
            kernel = tf.gather(kernel, filter_order, axis=0)
            kernel = torch.gather(kernel, filter_order, axis=0)
        filters = impulse_responses.gabor_filters(kernel, self._kernel_size)
        real_filters = tf.math.real(filters)
        img_filters = tf.math.imag(filters)
        stacked_filters = tf.stack([real_filters, img_filters], axis=1)
        stacked_filters = tf.reshape(stacked_filters,
                                     [2 * self._filters, self._kernel_size])
        stacked_filters = tf.expand_dims(
            tf.transpose(stacked_filters, perm=(1, 0)), axis=1)
        outputs = tf.nn.conv1d(
            inputs, stacked_filters, stride=self._strides, padding=self._padding)
        if self._use_bias:
            outputs = tf.nn.bias_add(outputs, self._bias, data_format='NWC')
        return outputs



class LEAF(nn.Module):
    def __init__(self):
        super(LEAF, self).__init__()

