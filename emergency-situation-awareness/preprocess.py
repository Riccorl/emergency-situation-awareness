import re
import string
from typing import Dict, List, Set, Tuple

import gensim
import numpy as np
from keras_preprocessing.text import maketrans
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TweetTokenizer

import utils


def build_vocab(data: List[List[str]]) -> Dict[str, int]:
    """
    Compute the vocab from the bigrams
    :param data: data set files
    :return: Dictionary from bigram to int
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for dataset in data:
        for sentence in dataset:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = len(vocab)
    return vocab


def vocab_from_w2v(word2vec: gensim.models.word2vec.Word2Vec) -> Dict[str, int]:
    """
    :param word2vec: trained Gensim Word2Vec model
    :return: a dictionary from token to int
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for index, word in enumerate(word2vec.wv.index2word):
        vocab[word] = index + 2
    return vocab


def clear_tweets(tweets: List[str]) -> List:
    """
    Clear list of tweets.
    :param tweets: list of tweets.
    :return: cleared tweets.
    """
    # preprocessing stuffs
    stops = set(stopwords.words("english")) | set(string.punctuation)
    html_regex = re.compile(r"^https?:\/\/.*[\r\n]*")
    tokenizer = TweetTokenizer()
    return [_clear_tweet(tweet, tokenizer, stops, html_regex) for tweet in tweets]


def _clear_tweet(tweet: str, tokenizer, stops: Set, html_regex) -> List:
    """
    Clean the tweet in input.
    :param tweet: tweet to clean.
    :param stops: set of stop words and punctuation to remove.
    :return: tweet cleaned.
    """
    return [word for word in tokenizer.tokenize(tweet.lower()) if word not in stops and not html_regex.search(word)]


def compute_x(
    features, vocab: Dict[str, int], max_len: int = 200, pad: bool = True
) -> np.ndarray:
    """
    Compute the features X.
    :param features: feature file.
    :param vocab: vocab.
    :param max_len: max len to pad.
    :param pad: If True pad the matrix, otherwise return the matrix not padded.
    :return: the feature vectors.
    """
    data = [
        [vocab[word] if word in vocab else vocab["<UNK>"] for word in l]
        for l in features
    ]
    if pad:
        return pad_sequences(data, truncating="post", padding="post", maxlen=max_len)
    else:
        return np.array(data)


def batch_generator(
    features: List[str],
    labels: List[str],
    vocab: Dict[str, int],
    batch_size: int = 32,
    max_input_len: int = 0,
):
    """
    Generates batches of data from features and labels,
    use it with Keras model.
    :param features: list of unigrams feature
    :param labels: list of labels
    :param vocab: unigram vocab
    :param batch_size: size of the batch to yield
    :param n_classes: number of classes
    :param max_input_len: max len of the input
    :return: processed features and labels, in batches
    """

    while True:
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            max_len = len(max(features[start:end], key=len))

            if max_input_len > 0:
                # truncate the sequence
                max_len = max_len if max_len < max_input_len else max_input_len

            x_batch = compute_x(features[start:end], vocab, max_len=max_len)
            y_batch = np.array(labels[start:end])
            yield x_batch, y_batch


def tf_idf_conversion(
    train_x: List[List[str]], max_features: int = None
) -> Tuple[List[List[int]], TfidfVectorizer]:
    """
    This method is used to map each sentences entry with tf-idf format
    :param train_x: input sentences
    :param max_features: number of maximun features to taking into account
    :return: coverted sentences, tf-idf distribution
    """

    train_x = utils.flatten(train_x)
    tfidf_vect = TfidfVectorizer(max_features=max_features, lowercase=False)
    train_x_tfidf = tfidf_vect.fit_transform(train_x)
    return train_x_tfidf, tfidf_vect
