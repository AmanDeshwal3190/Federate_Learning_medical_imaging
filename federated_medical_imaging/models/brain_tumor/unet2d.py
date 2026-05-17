"""
2D U-Net model for brain tumor segmentation from slices.
"""
import tensorflow as tf

def build_unet2d(input_shape=(240, 240, 4), num_classes=3, num_filters=[32, 64, 128, 256], dropout_rate=0.3):
    """
    Builds a standard 2D U-Net architectural model.
    
    Args:
        input_shape: Expected shape of 2D input slices.
        num_classes: Number of output channels (e.g. 3 for subregions or 4).
        num_filters: Base number of filters at each depth.
        dropout_rate: Dropout rate for regularization.
        
    Returns:
        Compiled tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    skip_connections = []
    x = inputs
    for idx, f in enumerate(num_filters):
        x = tf.keras.layers.Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        skip_connections.append(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2)(x)
        
    # Bottleneck
    x = tf.keras.layers.Conv2D(num_filters[-1]*2, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters[-1]*2, 3, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Decoder
    for idx, f in enumerate(reversed(num_filters)):
        x = tf.keras.layers.UpSampling2D(2)(x)
        skip = skip_connections[-(idx+1)]
        x = tf.keras.layers.Concatenate()([x, skip])
        
        x = tf.keras.layers.Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='UNet2D_BrainTumor')
    return model
