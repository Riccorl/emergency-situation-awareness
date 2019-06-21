from tensorflow import keras as k
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import (
    Dense,
    Embedding,
    Bidirectional,
    LSTM,
    Input,
    Dropout,
    TimeDistributed,
)


def build_model(
    layer: k.layers = LSTM,
    hidden_size: int = 256,
    input_length: int = None,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.1,
    learning_rate: float = 0.002,
    vocab_size: int = None,
    embedding_size: int = 300,
    train_embeddings: bool = False,
) -> Model:

    input_layer = Input(shape=(input_length,))
    em = Embedding(
        vocab_size, embedding_size, input_length=input_length, mask_zero=True
    )(input_layer)

    drop_em = Dropout(0.3)(em)

    lstm1 = Bidirectional(
        layer(
            units=hidden_size,
            dropout=0.3,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
        )
    )(drop_em)
    # lstm2 = Bidirectional(
    #     layer(units=hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout)
    # )(lstm1)

    output = Dense(1, activation="sigmoid")(lstm1)
    model = Model(inputs=input_layer, outputs=output)
    optimizer = k.optimizers.Adam()
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["acc"])
    model.summary()
    return model
