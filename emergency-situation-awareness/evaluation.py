import sklearn
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import config
import preprocess
import utils


def classification_report(y_test, y_pred, target):
    """
    Compute a classification report using sklearn.
    :param y_test: true labels.
    :param y_pred: predicted labels.
    :param target: label names.
    :return: a classification report from sklearn.
    """
    return sklearn.metrics.classification_report(y_test, y_pred, target_names=target)


def evaluate_keras(model) -> None:
    """
    Evaluate the model printing a classification report (acc, prec, recall and f1).
    :param model: the trained model.
    :return:
    """

    x_test, y_test = utils.load_datasets(
        config.CRISIS_EVAL_DIR, config.NORMAL_EVAL_DIR, limit=1000
    )
    x_test = preprocess.clear_tweets(x_test)
    vocab = utils.read_dictionary(config.TRAIN_VOCAB)
    x_test = preprocess.compute_x(x_test, vocab, max_len=100)[:, :, 0]
    y_pred = model.predict(x_test, batch_size=64)
    y_pred = [1 if y > 0.5 else 0 for y in y_pred]
    cr = classification_report(y_test, y_pred, ["normal", "crisis"])
    print("Classification report : \n", cr)


def evaluate_sklearn(model, tfidf_vec: TfidfVectorizer, kind_model: str = None) -> None:
    """
    This method is used to evaluate a sklearn model
    :param model: sklearn model (eg. SVM, naive_bayes)
    :param tfidf_vec: tf-idf distribution
    :param kind_model: a string used to print
    :return: None
    """
    x_test, y_test = utils.load_datasets(
        config.CRISIS_EVAL_DIR, config.NORMAL_EVAL_DIR, limit=1000
    )
    x_test = preprocess.clear_tweets(x_test)

    x_test = utils.flatten(x_test)
    x_test = tfidf_vec.transform(x_test)
    y_pred = model.predict(x_test)

    print(
        "Evaluating", kind_model if kind_model is not None else "Sklearn model", "..."
    )

    print("Accuracy Score:", accuracy_score(y_pred, y_test) * 100)
    cr = classification_report(y_test, y_pred, ["normal", "crisis"])
    print("Classification report : \n", cr)


if __name__ == "__main__":
    model = tf.keras.models.load_model(str(config.OUTPUT_DIR / "model.h5"))
    evaluate_keras(model)
