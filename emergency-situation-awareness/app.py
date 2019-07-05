import numpy as np

import config
import evaluation
import preprocess
import train
import utils


def preprocessing():
    print("Loading tweets...")
    features, labels = utils.load_datasets(
        config.CRISIS_TRAIN_DIR, config.NORMAL_TRAIN_DIR, limit=30000
    )
    print("Clearing tweets...")
    features = preprocess.clear_tweets(features)
    return features, labels


def main():
    np.random.seed(32)

    train_x, train_y = preprocessing()
    utils.plot_space(train_x[0:40] + train_x[300000:300040], train_y[0:40] + train_y[300000:300040])
    # print("Training Keras...")
    # model = train.train_keras(train_x, train_y, config.CRISIS_PRE_TRAINED)
    # print("Evaluate Keras...")
    # evaluation.evaluate_keras(model)

    print("Training Naive Bayes...")
    model, tfidf_vec = train.train_bayes(train_x, train_y)
    print("Evaluate Bayes...")
    evaluation.evaluate_sklearn(model, tfidf_vec, kind_model="Naive Bayes")

    # print("Training SVM...")
    # model, tfidf_vec = train.train_svm(train_x, train_y)
    # print("Evaluate SVM...")
    # evaluation.evaluate_sklearn(model, tfidf_vec, kind_model="SVM")


if __name__ == "__main__":
    main()
