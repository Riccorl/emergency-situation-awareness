import gensim
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (
    Dense,
    Embedding,
    Bidirectional,
    LSTM,
    Input,
    Layer,
    SpatialDropout1D
)


def build_model(
    layer: keras.layers = LSTM,
    hidden_size: int = 256,
    input_length: int = None,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.1,
    learning_rate: float = 0.002,
    vocab_size: int = None,
    word2vec: gensim.models.word2vec.Word2Vec = None,
    embedding_size: int = 300,
    train_embeddings: bool = False,
) -> Model:

    input_layer = Input(shape=(input_length,))
    if word2vec:
        em = get_keras_embedding(word2vec, train_embeddings)(input_layer)
    else:
        em = Embedding(
            vocab_size, embedding_size, input_length=input_length, mask_zero=True
        )(input_layer)
    # em = ELMoEmbedding(idx2word=idx2word, output_mode="default", trainable=True)(input_layer)
    dr = SpatialDropout1D(0.6)(em)
    lstm1 = Bidirectional(
        layer(
            units=hidden_size,
            # dropout=dropout,
            # recurrent_dropout=recurrent_dropout,
            return_sequences=False,
        )
    )(dr)
    # lstm2 = Bidirectional(
    #     layer(units=hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout)
    # )(lstm1)

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
        mask_zero=False,
        trainable=trainable,
    )


class ELMoEmbedding(Layer):
    def __init__(self, idx2word, output_mode="default", trainable=True, **kwargs):
        assert output_mode in [
            "default",
            "word_emb",
            "lstm_outputs1",
            "lstm_outputs2",
            "elmo",
        ]
        assert trainable in [True, False]
        self.idx2word = idx2word
        self.output_mode = output_mode
        self.trainable = trainable
        self.max_length = None
        self.word_mapping = None
        self.lookup_table = None
        self.elmo_model = None
        self.embedding = None
        super(ELMoEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.max_length = input_shape[1]
        self.word_mapping = [
            x[1] for x in sorted(self.idx2word.items(), key=lambda x: x[0])
        ]
        self.lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(
            self.word_mapping, default_value="<UNK>"
        )
        self.lookup_table.init.run(session=K.get_session())
        self.elmo_model = tf_hub.Module(
            "https://tfhub.dev/google/elmo/2", trainable=self.trainable
        )
        super(ELMoEmbedding, self).build(input_shape)

    def call(self, x, **kwargs):
        x = tf.cast(x, dtype=tf.int64)
        sequence_lengths = tf.cast(tf.count_nonzero(x, axis=1), dtype=tf.int32)
        strings = self.lookup_table.lookup(x)
        inputs = {"tokens": strings, "sequence_len": sequence_lengths}
        return self.elmo_model(inputs, signature="tokens", as_dict=True)[
            self.output_mode
        ]

    def compute_output_shape(self, input_shape):
        if self.output_mode == "default":
            return input_shape[0], 1024
        if self.output_mode == "word_emb":
            return input_shape[0], self.max_length, 512
        if self.output_mode == "lstm_outputs1":
            return input_shape[0], self.max_length, 1024
        if self.output_mode == "lstm_outputs2":
            return input_shape[0], self.max_length, 1024
        if self.output_mode == "elmo":
            return input_shape[0], self.max_length, 1024

    def get_config(self):
        config = {"idx2word": self.idx2word, "output_mode": self.output_mode}
        return list(config.items())
