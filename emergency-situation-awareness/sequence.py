from typing import Dict

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import Sequence


class TextSequence(Sequence):
    def __init__(
        self, x_set, y_set, batch_size, vocab, max_len, num_classes: int = None
    ):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.compute_x(
            self.x[idx * self.batch_size : (idx + 1) * self.batch_size],
            self.vocab,
            max_len=self.max_len,
        )
        batch_y = np.array(self.y[idx * self.batch_size : (idx + 1) * self.batch_size])

        return batch_x, batch_y

    def compute_x(
        self, features, vocab: Dict[str, int], max_len: int = 200, pad: bool = True
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
            return pad_sequences(
                data, truncating="post", padding="post", maxlen=max_len
            )
        else:
            return np.array(data)
