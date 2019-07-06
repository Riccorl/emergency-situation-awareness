import gensim
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import (
    Dense,
    Embedding,
    Bidirectional,
    LSTM,
    Input,
)


def build_model(
    layer: keras.layers = LSTM,
    hidden_size: int = 256,
    input_length: int = None,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.1,
    word2vec: gensim.models.word2vec.Word2Vec = None,
    train_embeddings: bool = False,
) -> Model:

    input_layer = Input(shape=(input_length,))
    em = get_keras_embedding(word2vec, train_embeddings)(input_layer)

    lstm1 = Bidirectional(
        layer(
            units=hidden_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
        )
    )(em)

    dense = Dense(100, activation="relu")(lstm1)
    output = Dense(1, activation="sigmoid")(dense)
    model = Model(inputs=input_layer, outputs=output)
    optimizer = keras.optimizers.Adam()
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["acc"])
    model.summary()
    return model


def get_keras_embedding(
    word2vec: gensim.models.word2vec.Word2Vec, trainable: bool = False
) -> Embedding:
    """
    Return a Tensorflow Keras 'Embedding' layer with weights set as the
    Word2Vec model's learned word embeddings.
    :param word2vec: gensim Word2Vec model
    :param trainable: if False, the weights are frozen and stopped from being updated.
                      If True, the weights can/will be further trained/updated.
    :return: a tf.keras.layers.Embedding layer.
    """
    weights = word2vec.wv.vectors
    # random vector for pad
    pad = np.random.rand(1, weights.shape[1])
    # mean vector for unknowns
    unk = np.mean(weights, axis=0, keepdims=True)
    weights = np.concatenate((pad, unk, weights))

    return Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        mask_zero=True,
        trainable=trainable,
    )
