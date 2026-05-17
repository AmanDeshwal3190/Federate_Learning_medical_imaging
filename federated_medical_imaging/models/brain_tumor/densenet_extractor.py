"""
DenseNet-121 feature extractor for brain tumor classification.
Uses pretrained DenseNet-121 with custom head for medical image features.
Also implements GoogleNet Inception-based multiscale feature extraction.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
import numpy as np
from typing import Tuple, List


def build_densenet_extractor(input_shape: Tuple[int, ...] = (224, 224, 3),
                               num_features: int = 1024) -> Model:
    """
    Build DenseNet-121 feature extractor.
    
    Architecture:
    1. Load DenseNet-121 pretrained on ImageNet (include_top=False)
    2. Add GlobalAveragePooling2D
    3. Add Dense(num_features, activation='relu')
    4. Add BatchNormalization
    5. Add Dropout(0.5)
    
    The model outputs a feature vector of size num_features.
    When used for grayscale medical images (1 channel), the input is
    replicated 3x to match DenseNet's expected 3-channel input.
    
    Args:
        input_shape: Input image shape (default (224, 224, 3) for pretrained)
        num_features: Size of output feature vector
    Returns:
        Keras Model that outputs feature vectors
    """
    inputs = layers.Input(shape=input_shape)
    
    if input_shape[-1] == 1:
        x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])
    else:
        x = inputs
        
    base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = True # Allow fine-tuning
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_features, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    return Model(inputs=inputs, outputs=x, name="densenet_extractor")


class InceptionMultiscaleBlock(layers.Layer):
    """
    GoogleNet Inception-inspired multiscale feature extraction block.
    
    Extracts features at multiple scales using parallel convolution paths:
    - Path 1: 1x1 Conv
    - Path 2: 1x1 Conv → 3x3 Conv
    - Path 3: 1x1 Conv → 5x5 Conv
    - Path 4: MaxPool → 1x1 Conv
    
    All paths are concatenated along the channel dimension.
    """
    
    def __init__(self, filters_1x1: int = 64, filters_3x3: int = 128,
                 filters_5x5: int = 32, filters_pool: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.filters_1x1 = filters_1x1
        self.filters_3x3 = filters_3x3
        self.filters_5x5 = filters_5x5
        self.filters_pool = filters_pool
        
        # Path 1: 1x1
        self.path1_1x1 = layers.Conv2D(filters_1x1, 1, padding='same', activation='relu')
        
        # Path 2: 1x1 -> 3x3
        self.path2_1x1 = layers.Conv2D(filters_3x3 // 2, 1, padding='same', activation='relu')
        self.path2_3x3 = layers.Conv2D(filters_3x3, 3, padding='same', activation='relu')
        
        # Path 3: 1x1 -> 5x5
        self.path3_1x1 = layers.Conv2D(filters_5x5 // 2, 1, padding='same', activation='relu')
        self.path3_5x5 = layers.Conv2D(filters_5x5, 5, padding='same', activation='relu')
        
        # Path 4: MaxPool -> 1x1
        self.path4_pool = layers.MaxPooling2D(3, strides=1, padding='same')
        self.path4_1x1 = layers.Conv2D(filters_pool, 1, padding='same', activation='relu')
        
    def call(self, inputs, training=None):
        p1 = self.path1_1x1(inputs)
        
        p2 = self.path2_1x1(inputs)
        p2 = self.path2_3x3(p2)
        
        p3 = self.path3_1x1(inputs)
        p3 = self.path3_5x5(p3)
        
        p4 = self.path4_pool(inputs)
        p4 = self.path4_1x1(p4)
        
        return layers.Concatenate(axis=-1)([p1, p2, p3, p4])
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters_1x1": self.filters_1x1,
            "filters_3x3": self.filters_3x3,
            "filters_5x5": self.filters_5x5,
            "filters_pool": self.filters_pool
        })
        return config


class SpatialAttention(layers.Layer):
    """
    Spatial attention mechanism to focus on tumor regions.
    
    1. Compute channel-wise average and max pooling
    2. Concatenate the two maps
    3. Apply Conv2D(1, kernel_size=7, activation='sigmoid')
    4. Multiply with input feature map
    """
    
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1, kernel_size=kernel_size, padding='same', activation='sigmoid', use_bias=False)
        
    def call(self, inputs, training=None):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv(concat)
        return inputs * attention
        
    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class SelfAttention(layers.Layer):
    """
    Self-attention mechanism for capturing long-range dependencies.
    
    1. Compute Query, Key, Value projections
    2. Compute attention weights: softmax(Q @ K^T / sqrt(d_k))
    3. Apply attention to Values
    """
    
    def __init__(self, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)
        
    def call(self, inputs, training=None):
        shape = tf.shape(inputs)
        batch = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]
        
        # Reshape to sequence (batch, H*W, channels)
        x_seq = tf.reshape(inputs, [batch, h * w, c])
        
        # Apply multi-head attention (self attention)
        attention_out = self.mha(x_seq, x_seq, x_seq, training=training)
        
        # Reshape back to image
        return tf.reshape(attention_out, [batch, h, w, c])
        
    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config


def build_multiscale_feature_extractor(input_shape: Tuple[int, ...] = (224, 224, 1),
                                        num_features: int = 512) -> Model:
    """
    Build complete multiscale feature extraction model with attention.
    
    Architecture:
    Input → InceptionMultiscaleBlock → SpatialAttention → SelfAttention → 
    GlobalAveragePool → Dense(num_features)
    
    Args:
        input_shape: (224, 224, 1) for grayscale medical images
        num_features: Output feature vector dimensionality
    Returns:
        Keras Model outputting feature vectors
    """
    inputs = layers.Input(shape=input_shape)
    
    x = InceptionMultiscaleBlock()(inputs)
    x = SpatialAttention()(x)
    x = SelfAttention()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_features, activation='relu')(x)
    
    return Model(inputs=inputs, outputs=outputs, name="multiscale_extractor")


def extract_features(model: Model, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from all images in a dataset using the feature extractor.
    
    Args:
        model: Feature extraction model
        dataset: tf.data.Dataset yielding (images, labels)
    Returns:
        Tuple of (features_array, labels_array)
        - features_array: shape (N, num_features)
        - labels_array: shape (N,)
    """
    all_features = []
    all_labels = []
    
    for images, labels in dataset:
        features = model.predict(images, verbose=0)
        all_features.append(features)
        
        # Check if labels are one-hot encoded or class indices
        labels_np = labels.numpy()
        if len(labels_np.shape) > 1 and labels_np.shape[-1] > 1:
            lbls = np.argmax(labels_np, axis=-1)
        else:
            lbls = labels_np
            
        all_labels.append(lbls)
        
    if len(all_features) == 0:
        return np.array([]), np.array([])
        
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)
