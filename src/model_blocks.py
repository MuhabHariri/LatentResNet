import tensorflow as tf
from tensorflow.keras import layers
from src.layers import ECALayer, WeightedResidualConnection
from src.config import hidden_units
from src.config import DEEPRESNET_CONFIG, hidden_units, CLASSIFIER_DROPOUT_RATE


def DeepResNet_Unit(input_tensor, num_filters, kernel_size, dropout_rate,  strides=(1, 1)):
    """A single block: depthwise + pointwise + BN + GELU + dropout + ECA."""
    x = layers.DepthwiseConv2D(kernel_size, padding='same', strides=strides)(input_tensor)
    x = layers.Conv2D(num_filters, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = ECALayer(num_filters)(x)
    return x

def DeepResNetBlock(
    x,
    num_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    additional_layers=2,
    dropout_rate=0.10,
    residual_dropout_rate=0.025
):
    """A deeper ResNet-inspired block with grouped convolutions, ECA, spatial dropout, and GELU activation."""

    residual = x

    x = DeepResNet_Unit(x, num_filters, kernel_size, dropout_rate, strides)

    #Additional conv blocks
    for _ in range(additional_layers + 1):  # +1 to include the "final" one
        x = DeepResNet_Unit(x, num_filters, kernel_size, dropout_rate)

    #Residual path with optional dropout
    residual = layers.SpatialDropout2D(residual_dropout_rate)(residual)

    if residual.shape[-1] != num_filters or strides != (1, 1):
        residual = layers.Conv2D(num_filters, (1, 1), strides=strides, padding='same')(residual)
        residual = layers.BatchNormalization()(residual)

    x = WeightedResidualConnection()(x, residual)
    x = layers.Activation('gelu')(x)

    return x




def EncodDeepResNet_Model(input_tensor, num_classes):
    """
    Builds the deep ResNet-style model with multiple stages and a classification head.
    """

    # ----------- Stage 1 -----------
    cfg = DEEPRESNET_CONFIG["stage1"]
    x = DeepResNetBlock(
        input_tensor,
        **{k: cfg[k] for k in cfg if not k.startswith("residual_dropout_rate")},
        residual_dropout_rate=cfg["residual_dropout_rate_1"]
    )
    x = DeepResNetBlock(
        x,
        **{k: cfg[k] for k in cfg if not k.startswith("residual_dropout_rate")},
        residual_dropout_rate=cfg["residual_dropout_rate_2"]
    )
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # ----------- Stage 2 -----------
    cfg = DEEPRESNET_CONFIG["stage2"]
    x = DeepResNetBlock(
        x,
        **{k: cfg[k] for k in cfg if not k.startswith("residual_dropout_rate")},
        residual_dropout_rate=cfg["residual_dropout_rate_1"]
    )
    x = DeepResNetBlock(
        x,
        **{k: cfg[k] for k in cfg if not k.startswith("residual_dropout_rate")},
        residual_dropout_rate=cfg["residual_dropout_rate_2"]
    )
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # ----------- Stage 3 -----------
    cfg = DEEPRESNET_CONFIG["stage3"]
    x = DeepResNetBlock(
        x,
        **{k: cfg[k] for k in cfg if not k.startswith("residual_dropout_rate")},
        residual_dropout_rate=cfg["residual_dropout_rate_1"]
    )
    x = DeepResNetBlock(
        x,
        **{k: cfg[k] for k in cfg if not k.startswith("residual_dropout_rate")},
        residual_dropout_rate=cfg["residual_dropout_rate_2"]
    )

    # ----------- Classification Head -----------
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(CLASSIFIER_DROPOUT_RATE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)

    output = mlp_head(x, hidden_units, num_classes)

    return output




def mlp_head(x, hidden_units, output_units, activation=tf.keras.activations.gelu, dropout_rate=0.2):
    for units in hidden_units:
        x = layers.Dense(units, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(output_units, activation='softmax')(x)
    return x


