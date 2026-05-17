"""
3D U-Net with Squeeze-and-Excitation ResNet encoder for brain tumor segmentation.

Architecture:
- Encoder: 5 levels with SE-ResNet blocks
  Level 1: Conv3D(32) → SE-ResBlock → MaxPool3D
  Level 2: Conv3D(64) → SE-ResBlock → MaxPool3D
  Level 3: Conv3D(128) → SE-ResBlock → MaxPool3D
  Level 4: Conv3D(256) → SE-ResBlock → MaxPool3D
  Level 5 (bottleneck): Conv3D(512) → SE-ResBlock

- Decoder: 4 levels with skip connections
  Level 4: UpConv3D(256) + skip4 → Conv3D(256) → Conv3D(256)
  Level 3: UpConv3D(128) + skip3 → Conv3D(128) → Conv3D(128)
  Level 2: UpConv3D(64) + skip2 → Conv3D(64) → Conv3D(64)
  Level 1: UpConv3D(32) + skip1 → Conv3D(32) → Conv3D(32)

- Output: Conv3D(num_classes, 1x1x1) → Softmax

All Conv3D use:
- kernel_size=(3,3,3), padding='same'
- BatchNormalization after each Conv3D
- ReLU activation
- Dropout(0.3) in encoder

SE Block (Squeeze-and-Excitation):
- GlobalAveragePooling3D
- Dense(filters // reduction_ratio) → ReLU
- Dense(filters) → Sigmoid
- Multiply with input feature map
- reduction_ratio = 16
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import List, Tuple


class SqueezeExcitation3D(layers.Layer):
    """3D Squeeze-and-Excitation block."""
    
    def __init__(self, filters: int, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.reduction_ratio = reduction_ratio
        
        self.global_pool = layers.GlobalAveragePooling3D()
        self.reshape = layers.Reshape((1, 1, 1, filters))
        self.dense1 = layers.Dense(max(1, filters // reduction_ratio), activation='relu', use_bias=False)
        self.dense2 = layers.Dense(filters, activation='sigmoid', use_bias=False)
    
    def call(self, inputs, training=None):
        x = self.global_pool(inputs)
        x = self.reshape(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return inputs * x
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "reduction_ratio": self.reduction_ratio
        })
        return config


class SEResBlock3D(layers.Layer):
    """3D Residual block with Squeeze-and-Excitation."""
    
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
        self.conv1 = layers.Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        
        self.conv2 = layers.Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        
        self.se_block = SqueezeExcitation3D(filters)
        
        self.relu2 = layers.Activation('relu')
        
        self.shortcut_conv = layers.Conv3D(filters, 1, padding='same', kernel_initializer='he_normal')
        self.shortcut_bn = layers.BatchNormalization()
    
    def build(self, input_shape):
        self.needs_projection = input_shape[-1] != self.filters
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = self.se_block(x, training=training)
        
        if self.needs_projection:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
            
        x = layers.Add()([x, shortcut])
        x = self.relu2(x)
        return x
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters
        })
        return config


def build_unet3d(input_shape: Tuple[int, ...] = (128, 128, 128, 4),
                  num_classes: int = 4,
                  num_filters: List[int] = [32, 64, 128, 256, 512],
                  dropout_rate: float = 0.3) -> Model:
    """
    Build complete 3D U-Net with SE-ResNet encoder.
    
    Args:
        input_shape: (128, 128, 128, 4) — 4 MRI modalities
        num_classes: 4 (Background, NCR/NET, Edema, ET)
        num_filters: Filter counts per encoder level
        dropout_rate: Dropout probability
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    skips = []
    x = inputs
    for i, filters in enumerate(num_filters[:-1]):
        x = layers.Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = SEResBlock3D(filters)(x)
        skips.append(x)
        
        x = layers.Dropout(dropout_rate)(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        
    # Bottleneck
    bottleneck_filters = num_filters[-1]
    x = layers.Conv3D(bottleneck_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = SEResBlock3D(bottleneck_filters)(x)
    
    # Decoder
    skips = list(reversed(skips))
    decoder_filters = list(reversed(num_filters[:-1]))
    
    for i, filters in enumerate(decoder_filters):
        x = layers.Conv3DTranspose(filters, 2, strides=(2, 2, 2), padding='same')(x)
        x = layers.Concatenate()([x, skips[i]])
        
        x = layers.Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv3D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
    # Output layer
    outputs = layers.Conv3D(num_classes, 1, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs, name="unet3d_se_resnet")


def dice_loss_tf(y_true, y_pred, smooth=1e-7):
    """
    Dice loss for TensorFlow/Keras training.
    Handles multi-class segmentation with one-hot encoded labels.
    
    DiceLoss = 1 - (2 * sum(y_true * y_pred) + smooth) / (sum(y_true) + sum(y_pred) + smooth)
    
    Computed per-class and averaged.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Sum over spatial dimensions D, H, W (axes 1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    y_true_sum = tf.reduce_sum(y_true, axis=(1, 2, 3))
    y_pred_sum = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    
    dice = (2.0 * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
    
    # Dice loss per class
    dice_loss_per_class = 1.0 - dice
    
    # Average over classes, then over batch
    return tf.reduce_mean(dice_loss_per_class)


def combined_loss(y_true, y_pred):
    """Combined Dice loss + Categorical Cross-Entropy for stable training."""
    dice = dice_loss_tf(y_true, y_pred)
    bce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # Ensure BCE has the same shape/reduction (categorical_crossentropy reduces the last axis)
    # Resulting shape for BCE is (batch, D, H, W). We need to mean it across D, H, W.
    bce = tf.reduce_mean(bce, axis=(1, 2, 3))
    return dice + tf.reduce_mean(bce)

