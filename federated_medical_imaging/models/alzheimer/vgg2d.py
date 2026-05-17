"""
2D VGG-style model for Alzheimer's Disease classification from 2D images.
"""
import tensorflow as tf

def build_vgg2d(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.5):
    """
    Builds a simple 2D VGG-style CNN for image classification.
    
    Args:
        input_shape: Expected shape of the 2D images.
        num_classes: Number of prediction classes.
        dropout_rate: Dropout rate for dense layers.
        
    Returns:
        Compiled tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Classification Head
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='VGG2D_Alzheimer')
    return model
