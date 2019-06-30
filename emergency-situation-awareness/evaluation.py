import gensim
import sklearn
from tqdm import tqdm

import utils
import preprocess
import config
import tensorflow as tf


def classification_report(y_test, y_pred, target):
    """
    Compute a classification report using sklearn.
    :param y_test: true labels.
    :param y_pred: predicted labels.
    :param target: label names.
    :return: a classification report from sklearn.
    """
    return sklearn.metrics.classification_report(
        y_test, y_pred, target_names=target
    )


def evaluate(model):
    """
    Evaluate the model printing a classification report (acc, prec, recall and f1).
    :param model: the trained model.
    :return:
    """

    x_test, y_test = utils.load_datasets(config.CRISIS_EVAL_DIR, config.NORMAL_EVAL_DIR, limit=1000)
    x_test = preprocess.clear_tweets(x_test)
    vocab = utils.read_dictionary(config.TRAIN_VOCAB)
    x_test = preprocess.compute_x(x_test, vocab, max_len=100)[:, :, 0]
    y_pred = model.predict(x_test, batch_size=64)
    y_pred = [1 if y > 0.5 else 0 for y in y_pred]
    cr = classification_report(y_test, y_pred, ["normal", "crisis"])
    print("Classification report : \n", cr)


if __name__ == '__main__':
    model = tf.keras.models.load_model(str(config.OUTPUT_DIR / "model.h5"))
    evaluate(model)
