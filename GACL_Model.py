import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout, LSTM, Bidirectional, Dense,
    Attention, GlobalAveragePooling1D, concatenate, LayerNormalization,
    MultiHeadAttention, Add, Flatten, Dense, Reshape)
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder


# Define Multi-Scale Convolutional Block
def multi_scale_conv_block(inputs):
    conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    conv2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(inputs)
    conv3 = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')(inputs)

    concat = concatenate([conv1, conv2, conv3], axis=-1)
    bn = BatchNormalization()(concat)
    return Dropout(0.3)(bn)

# Attention Fusion Layer
def attention_fusion(inputs):
    # Spatial, Temporal, and Frequency Attention
    spatial_attention = Attention()([inputs, inputs])
    temporal_attention = Attention()([spatial_attention, spatial_attention])

    # Combining the attentions
    fused_attention = Add()([spatial_attention, temporal_attention])
    return fused_attention

def graph_conv_layer(inputs):
    conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    bn = BatchNormalization()(conv)
    return Dropout(0.3)(bn)


# Bi-Directional LSTM with Attention
def bi_lstm_with_attention(inputs):
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_attention = Attention()([lstm_out, lstm_out])
    return lstm_attention

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # Multi-Scale Convolutional Block
    x = multi_scale_conv_block(inputs)

    # Attention Fusion Layer
    x = attention_fusion(x)

    # Graph Convolutional Layer
    x = graph_conv_layer(x)

    # Bi-Directional LSTM with Time-Distorted Attention
    x = bi_lstm_with_attention(x)

    # Hierarchical Feature Aggregation and Dense Layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)

    # Output Layer
    outputs = Dense(2, activation='softmax')(x)  # Binary classification

    from tensorflow.keras.optimizers import Adam
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=1e-4)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model