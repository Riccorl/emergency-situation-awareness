from typing import Dict, List

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


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


def texts_to_sequences(lines: List[List[str]], word_index: Dict[str, int]):
    """
    Transforms each text to a sequence of integers.
    :param lines: A list of list of strings.
    :param word_index: dictionary of word indexes.
    :return: A list of sequences.
    """
    return [
        np.array([word_index.get(word) for word in line if word_index.get(word)])
        for line in lines
    ]


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

            X_batch = compute_x(features[start:end], vocab, max_len=max_len)
            y_batch = np.array(labels[start:end])
            yield X_batch, y_batch
