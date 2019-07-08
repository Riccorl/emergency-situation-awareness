import time
from typing import List, Tuple, Dict, Optional

import gensim
import numpy as np
import sklearn
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn import naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    make_scorer,
)
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold

import config
import models
import preprocess
import utils
from sequence import TextSequence


def train_keras(
    features: List[str],
    labels: List[int],
    path_embeddings: str,
    epochs: int = 5,
    hidden_size: int = 100,
    batch_size: int = 512,
) -> models:
    """
    This method is used to train the keras method
    :param features: input train
    :param labels: label train
    :param path_embeddings: path of pre-trained embeddings
    :param epochs: number of epochs
    :param hidden_size: number of hidden size
    :param batch_size: number of batch size
    :return: keras model
    """

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config_tf))

    tweets_tr, tweets_dev, labels_tr, labels_dev = train_test_split(
        features, labels, test_size=0.20
    )
    vocab, w2v, w2v_vocab = _process_keras(tweets_tr, labels_tr, path_embeddings)

    train_gen = TextSequence(
        tweets_tr, labels_tr, vocab=w2v_vocab, batch_size=batch_size, max_len=15
    )

    dev_gen = TextSequence(
        tweets_dev, labels_dev, vocab=w2v_vocab, batch_size=batch_size, max_len=15
    )

    model = models.build_model(
        hidden_size=hidden_size, dropout=0.6, recurrent_dropout=0.2, word2vec=w2v
    )

    cp_path = "model_chkpt.h5"
    cp = tf.keras.callbacks.ModelCheckpoint(
        cp_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, mode="min", verbose=1, restore_best_weights=True
    )

    print("Starting training...")
    start = time.time()
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(tweets_tr) // batch_size,
        epochs=epochs,
        validation_data=dev_gen,
        validation_steps=len(tweets_dev) // batch_size,
        shuffle=True,
        callbacks=[cp, es],
    )
    utils.plot_keras(history)
    end = time.time()
    print(utils.timer(start, end))
    model.save(str(config.OUTPUT_DIR / "model_big.h5"))
    print("Training complete.")

    return model


def _process_keras(
    features_train: List[str], labels: List[int], path_embeddings: str
) -> Tuple[Dict[str, int], Optional[Word2Vec], Dict[str, int]]:
    """
    This method is used to process the sentences with respect to a keras model
    :param features_train: input train
    :param labels: label set.
    :param path_embeddings: path of pre-trained embeddings
    :return: vocabulary, word2vec model, word2vec vocabulary
    """
    print("Loading pre-trained embeddings...")
    # load the w2v matrix with genism
    w2v = gensim.models.KeyedVectors.load_word2vec_format(path_embeddings, binary=True)
    for i, t in enumerate(features_train):
        if not t:
            del features_train[i]
            del labels[i]
    # build the vocab from the w2v model
    w2v_vocab = preprocess.vocab_from_w2v(w2v)
    print("Word2Vec model vocab len:", len(w2v_vocab))
    # build vocab from the dataset
    data_vocab = preprocess.build_vocab([features_train])
    # filter pretrained w2v with words from the dataset
    w2v = utils.restrict_w2v(w2v, set(data_vocab.keys()))
    w2v_vocab = preprocess.vocab_from_w2v(w2v)
    utils.write_dictionary(config.TRAIN_VOCAB, w2v_vocab)
    print("Cleaned vocab len:", len(w2v_vocab))
    # idx2word = {v: k for k, v in vocab.items()}
    return data_vocab, w2v, w2v_vocab


def train_bayes(train_x: List[List[str]], train_y: List[int]):
    """
    This method is used to train a naive bayes classifier
    :param train_x: input train
    :param train_y: label train
    :return: naive bayes model, tf-idf distribution
    """
    train_x, train_y, tfidf_vec = _process_linear(train_x, train_y, max_features=6000)
    naive = naive_bayes.MultinomialNB()
    print("\nCross Validation...")
    _train_linear(naive, train_x, train_y)
    print("\nFitting...")
    naive.fit(train_x, train_y)
    return naive, tfidf_vec


def train_svm(
    train_x: List[List[str]], train_y: List[int], c: float = 1.0, max_iter: int = 1000
):
    """
    This method is used to train a SVM classifier
    :param train_x: input train
    :param train_y: label train
    :param c: penalty parameter C of the error term.
    :param max_iter: number of maximum iteration
    :return: SVM model, tf-idf distribution
    """

    train_x, train_y, tfidf_vec = _process_linear(train_x, train_y, max_features=1000)
    print(train_x.shape)
    svm_model = svm.LinearSVC(C=c, max_iter=max_iter, dual=False)
    # svm_model = svm.SVC(
    #     kernel="rbf", gamma="scale", degree=3, max_iter=max_iter, C=c, verbose=False
    # )
    print("Cross Validation...")
    _train_linear(svm_model, train_x, train_y)
    print("\nFitting...")
    svm_model.fit(train_x, train_y)
    return svm_model, tfidf_vec


def _process_linear(
    train_x: List[List[str]], train_y: List[int], max_features: int = None
) -> Tuple[np.array, List[int], TfidfVectorizer]:
    """
    This method is used to process the sentences with respect to a sklearn model
    :param train_x: input train
    :return: tf-idf sentences conversion, tf-idf distribution
    """
    train_x, train_y = sklearn.utils.shuffle(train_x, train_y)
    train_x, tfidf_vect = preprocess.tf_idf_conversion(train_x, max_features)
    return train_x, train_y, tfidf_vect


def _train_linear(model, train_x, train_y):
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, pos_label=1),
        "recall": make_scorer(recall_score, pos_label=1),
        "f1": make_scorer(fbeta_score, beta=1, pos_label=1),
    }
    start = time.time()
    kfold = StratifiedKFold(10, True, 1)
    accuracy, precision, recall, fscore = validation(
        train_x, train_y, model, kfold, scoring
    )
    end = time.time()
    print("Accuracy: {0:.2f}".format(accuracy))
    print("Precision: {0:.2f}".format(precision))
    print("Recall: {0:.2f}".format(recall))
    print("F1 score: {0:.2f}".format(fscore))
    print("Execution Time: " + utils.timer(start, end))
    print("")
    print("Plotting learning curve...")
    utils.plot_learning_curve(train_x, train_y, model, kfold, scoring)
    # utils.plot_learning_curve(train_x, train_y, model, kfold, scoring["precision"])
    print("Done.")


def validation(train_x, train_y, estimator, cv, scoring):
    scores = cross_validate(
        estimator, train_x, train_y, cv=cv, scoring=scoring, n_jobs=-1
    )
    return (
        np.mean(scores["test_accuracy"]),
        np.mean(scores["test_precision"]),
        np.mean(scores["test_recall"]),
        np.mean(scores["test_f1"]),
    )
