import time

import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
import models
import utils
from sequence import TextSequence


def train_keras(
    features,
    labels,
    w2v,
    w2v_vocab,
    epochs: int = 5,
    hidden_size: int = 100,
    batch_size: int = 512,
):

    tweets_tr, tweets_dev, labels_tr, labels_dev = train_test_split(
        features, labels, test_size=0.10
    )

    train_gen = TextSequence(
        tweets_tr, labels_tr, vocab=w2v_vocab, batch_size=batch_size, max_len=100
    )

    dev_gen = TextSequence(
        tweets_dev, labels_dev, vocab=w2v_vocab, batch_size=batch_size, max_len=100
    )

    model = models.build_model(
        layer=tf.keras.layers.GRU,
        hidden_size=hidden_size,
        dropout=0.4,
        recurrent_dropout=0.2,
        vocab_size=len(w2v_vocab),
        word2vec=w2v,
    )

    print("Starting training...")
    start = time.time()
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(tweets_tr) // batch_size,
        epochs=epochs,
        validation_data=dev_gen,
        validation_steps=len(tweets_dev) // batch_size,
        shuffle=True
        # callbacks=[es, cp],
    )
    end = time.time()
    print(utils.timer(start, end))
    model.save(str(config.OUTPUT_DIR / "model.h5"))
    print("Training complete.")

    return model
