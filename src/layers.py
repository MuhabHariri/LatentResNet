import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models

class ECALayer(layers.Layer):
    """
    Efficient Channel Attention (ECA) Layer.

    This layer applies channel-wise attention using a 1D convolution over 
    global average pooled features. Kernel size is adaptively determined.
    """

    def __init__(self, channels, gamma=2, b=1, activation='sigmoid', **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.channels = channels
        self.gamma = gamma
        self.b = b
        self.activation = activation

        #Calculate adaptive kernel size (odd number)
        t = abs((np.log2(channels) + b) / gamma)
        k = int(t) if int(t) % 2 else int(t) + 1

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.conv = layers.Conv1D(
            filters=1,
            kernel_size=k,
            padding='same',
            use_bias=False,
            activation=activation
        )

    def call(self, inputs):
        #Squeeze: Global average pooling
        y = self.avg_pool(inputs)                        
        y = tf.expand_dims(y, axis=-1)                    

        #Excitation: 1D conv across channels
        y = self.conv(y)                                  
        y = tf.squeeze(y, axis=-1)                        

        #Reshape to apply channel-wise weights
        y = tf.reshape(y, [-1, 1, 1, self.channels])       

        #Scale the input
        return inputs * y

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'gamma': self.gamma,
            'b': self.b,
            'activation': self.activation,
        })
        return config


class WeightedResidualConnection(layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedResidualConnection, self).__init__(**kwargs)
        self.weight = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)  # Layer normalization

    def call(self, inputs, residual):
        #Scale the residual by the learned weight
        weighted_residual = residual * self.weight
        #Combine inputs and weighted residual
        combined = layers.add([inputs, weighted_residual])
        #Apply layer normalization
        normalized_output = self.layer_norm(combined)
        return normalized_output



def remove_spatial_dropout(encoder):
    inputs = tf.keras.Input(shape=encoder.input.shape[1:])
    x = inputs
    for layer in encoder.layers:
        if not isinstance(layer, layers.SpatialDropout2D):
            x = layer(x)
    return models.Model(inputs, x)






