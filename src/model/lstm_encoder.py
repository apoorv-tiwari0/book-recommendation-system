import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model


def masked_mean(tensor, mask):
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    tensor = tensor * mask
    return tf.reduce_sum(tensor, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)


def build_lstm_encoder(
    vocab_size: int,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    sequence_length: int = 300
):
    inputs = Input(shape=(sequence_length,), name="input_tokens")

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True
    )

    x = embedding_layer(inputs)
    mask = embedding_layer.compute_mask(inputs)

    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True)
    )(x)

    x = masked_mean(x, mask)

    embeddings = Dense(
        embedding_dim,
        activation=None,
        name="book_embedding"
    )(x)

    model = Model(inputs, embeddings, name="lstm_encoder")
    return model
