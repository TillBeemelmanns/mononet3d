from tensorflow import keras
from tensorflow.keras import layers


def small_resnet(input_shape):

    inputs = keras.Input(input_shape)
    x = layers.experimental.preprocessing.Resizing(256, 256)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    outputs = layers.GlobalMaxPooling2D()(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='alexnet_encoder')

    return model


def AlexNet(input_shape):

    inputs = keras.Input(input_shape)

    x = layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(256, (11, 11), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    outputs = layers.GlobalMaxPooling2D()(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='alexnet_encoder')

    return model