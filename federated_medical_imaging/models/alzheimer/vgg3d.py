"""
3D VGG-16 inspired CNN for Alzheimer's Disease classification.
4 convolutional blocks + 3 FC layers for binary classification (AD vs HC).
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from typing import Tuple, List, Optional


def build_conv_block(x, num_conv_layers: int, filters: int, block_name: str):
    """
    Build a convolutional block with multiple Conv3D layers followed by MaxPooling.
    
    Each Conv3D layer:
    - kernel_size=(3,3,3)
    - padding='same'
    - activation='relu'
    - kernel_initializer='he_normal'
    - kernel_regularizer=l2(1e-4)
    
    Followed by BatchNormalization after each Conv3D.
    Block ends with MaxPool3D(pool_size=(2,2,2)).
    
    Args:
        x: Input tensor
        num_conv_layers: Number of Conv3D layers (2 for blocks 1-2, 3 for blocks 3-4)
        filters: Number of filters
        block_name: Name prefix (e.g., "block1")
    Returns:
        Output tensor after the block
    """
    for i in range(num_conv_layers):
        x = layers.Conv3D(
            filters=filters,
            kernel_size=(3, 3, 3),
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"{block_name}_conv{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"{block_name}_bn{i+1}")(x)
        x = layers.Activation('relu', name=f"{block_name}_relu{i+1}")(x)
        
    x = layers.MaxPool3D(pool_size=(2, 2, 2), name=f"{block_name}_pool")(x)
    return x


def build_vgg3d(input_shape: Tuple[int, ...] = (182, 218, 182, 1),
                num_classes: int = 2,
                dropout_rate: float = 0.5) -> Model:
    """
    Build the 3D VGG-16 inspired model for AD classification.
    
    Architecture:
    Input(input_shape)
    → Block1: 2× Conv3D(64) + MaxPool3D
    → Block2: 2× Conv3D(128) + MaxPool3D
    → Block3: 3× Conv3D(256) + MaxPool3D
    → Block4: 3× Conv3D(512) + MaxPool3D
    → Flatten
    → Dropout(0.5)
    → Dense(128, ReLU)
    → Dropout(0.3)
    → Dense(64, ReLU)
    → Dropout(0.3)
    → Dense(2, Softmax)
    
    Total expected trainable parameters: ~5-10M depending on input shape.
    
    Args:
        input_shape: MRI volume shape with channel (H, W, D, 1)
                     ADNI: (182, 218, 182, 1)
                     OASIS: (176, 208, 176, 1)
        num_classes: 2 (AD=1, HC=0)
        dropout_rate: 0.5 for main dropout, 0.3 for intermediate
    Returns:
        Keras Model (uncompiled)
    """
    inputs = layers.Input(shape=input_shape, name="input_volume")
    
    # Block 1
    x = build_conv_block(inputs, num_conv_layers=2, filters=64, block_name="block1")
    
    # Block 2
    x = build_conv_block(x, num_conv_layers=2, filters=128, block_name="block2")
    
    # Block 3
    x = build_conv_block(x, num_conv_layers=3, filters=256, block_name="block3")
    
    # Block 4
    x = build_conv_block(x, num_conv_layers=3, filters=512, block_name="block4")
    
    # Classifier
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(dropout_rate, name="dropout_main")(x)
    
    x = layers.Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation('relu', name="fc1_relu")(x)
    x = layers.Dropout(0.3, name="dropout_fc1")(x)
    
    x = layers.Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Activation('relu', name="fc2_relu")(x)
    x = layers.Dropout(0.3, name="dropout_fc2")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="VGG3D")
    return model


def build_vgg3d_small(input_shape: Tuple[int, ...] = (96, 112, 96, 1),
                       num_classes: int = 2) -> Model:
    """
    Build a smaller version of VGG3D for faster training/debugging.
    Uses half the filters: [32, 64, 128, 256] instead of [64, 128, 256, 512].
    Input is downsampled to (96, 112, 96, 1).
    
    Use this for initial testing before full-scale training.
    """
    inputs = layers.Input(shape=input_shape, name="input_volume")
    
    # Block 1
    x = build_conv_block(inputs, num_conv_layers=2, filters=32, block_name="block1")
    
    # Block 2
    x = build_conv_block(x, num_conv_layers=2, filters=64, block_name="block2")
    
    # Block 3
    x = build_conv_block(x, num_conv_layers=3, filters=128, block_name="block3")
    
    # Block 4
    x = build_conv_block(x, num_conv_layers=3, filters=256, block_name="block4")
    
    # Classifier
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.5, name="dropout_main")(x)
    
    x = layers.Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation('relu', name="fc1_relu")(x)
    x = layers.Dropout(0.3, name="dropout_fc1")(x)
    
    x = layers.Dense(32, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Activation('relu', name="fc2_relu")(x)
    x = layers.Dropout(0.3, name="dropout_fc2")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="VGG3D_Small")
    return model
