import sklearn


def classification_report(y_test, y_pred, target):
    """

    :param y_test:
    :param y_pred:
    :param target:
    :return:
    """
    return sklearn.metrics.classification_report(
        y_test, y_pred, target_names=target
    )


def evaluate(model, x_test, y_test):
    """

    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    y_pred = model.predict(x_test)
    cr = classification_report(y_test, y_pred, ["normal", "crisis"])
    print("Classification report : \n", cr)
