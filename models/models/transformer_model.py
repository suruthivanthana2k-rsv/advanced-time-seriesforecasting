import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerBlock(layers.Layer):
    def _init_(self, head_size, num_heads, ff_dim, dropout=0.1):
        super()._init_()
        self.att = layers.MultiHeadAttention(num_heads, head_size)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(head_size),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

def build_transformer(seq_len, n_features):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = TransformerBlock(32, 4, 128)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model
