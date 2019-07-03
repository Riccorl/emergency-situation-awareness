import time
from typing import List, Tuple, Dict, Optional

import gensim
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn import naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer

import config
import models
import preprocess
import utils
from sequence import TextSequence


def train_keras(
    tweets_tr: List[str],
    tweets_dev: List[str],
    labels_tr: List[str],
    labels_dev: List[str],
    path_embeddings: str,
    epochs: int = 5,
    hidden_size: int = 100,
    batch_size: int = 512,
) -> models:
    """
    This method is used to train the keras method
    :param tweets_tr: input train
    :param tweets_dev: input development set
    :param labels_tr: label train
    :param labels_dev: label development set
    :param path_embeddings: path of pre-trained embeddings
    :param epochs: number of epochs
    :param hidden_size: number of hidden size
    :param batch_size: number of batch size
    :return: keras model
    """
    vocab, w2v, w2v_vocab = _process_keras(tweets_tr, tweets_dev, path_embeddings)

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


def _process_keras(
    features_train: List[str], features_dev: List[str], path_embeddings: str
) -> Tuple[Dict[str, int], Optional[Word2Vec], Dict[str, int]]:
    """
    This method is used to process the sentences with respect to a keras model
    :param features_train: input train
    :param features_dev: input development set
    :param path_embeddings: path of pre-trained embeddings
    :return: vocabulary, word2vec model, word2vec vocabulary
    """
    print("Loading pre-trained embeddings...")
    # load the w2v matrix with genism
    w2v = gensim.models.KeyedVectors.load_word2vec_format(path_embeddings, binary=True)
    # build the vocab from the w2v model
    w2v_vocab = preprocess.vocab_from_w2v(w2v)
    print("Word2Vec model vocab len:", len(w2v_vocab))
    # build vocab from the dataset
    data_vocab = preprocess.build_vocab([features_train, features_dev])
    # filter pretrained w2v with words from the dataset
    w2v = utils.restrict_w2v(w2v, set(data_vocab.keys()))
    w2v_vocab = preprocess.vocab_from_w2v(w2v)
    utils.write_dictionary(config.TRAIN_VOCAB, w2v_vocab)
    print("Cleaned vocab len:", len(w2v_vocab))
    # idx2word = {v: k for k, v in vocab.items()}
    return data_vocab, w2v, w2v_vocab


def train_bayes(train_x: List[List[str]], train_y: List[List[str]]):
    """
    This method is used to train a naive bayes classifier
    :param train_x: input train
    :param train_y: label train
    :return: naive bayes model, tf-idf distribution
    """

    train_x, tfidf_vec = _process_linear(train_x)

    naive = naive_bayes.MultinomialNB()
    naive.fit(train_x, train_y)

    return naive, tfidf_vec


def train_svm(
    train_x: List[List[str]],
    train_y: List[List[str]],
    c: int = 1.0,
    kernel: str = "linear",
    degree: int = 3,
    gamma: str = "auto",
    max_iter: int = 1000,
):
    """
    Thise method is used to train a SVM classifier
    :param train_x: input train
    :param train_y: label train
    :param c: penalty parameter C of the error term.
    :param kernel: specifies the kernel type to be used in the algorithm.
    :param degree: degree of the polynomial kernel function
    :param gamma: kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    :param max_iter: number of maximum iteration
    :return: SVM model, tf-idf distribution
    """

    train_x, tfidf_vec = _process_linear(train_x)
    svm_model = svm.SVC(
        C=c, kernel=kernel, degree=degree, gamma=gamma, max_iter=max_iter, verbose=True
    )
    svm_model.fit(train_x, train_y)
    return svm_model, tfidf_vec


def _process_linear(
    train_x: List[List[str]]
) -> Tuple[List[List[int]], TfidfVectorizer]:
    """
    This method is used to process the sentences with respect to a sklearn model
    :param train_x: input train
    :return: tf-idf sentences conversion, tf-idf distribution
    """
    return preprocess.tf_idf_conversion(train_x)
