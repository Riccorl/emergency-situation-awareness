import numpy as np

import config
import evaluation
import preprocess
import train
import utils


def visualizzation():
    print("Loading tweets...")
    features, labels = utils.load_datasets(
        config.CRISIS_TRAIN_DIR, config.NORMAL_TRAIN_DIR, limit=30
    )
    print("Clearing tweets...")
    features = preprocess.clear_tweets(features)
    return features, labels


def preprocessing(limit: int = 30000):
    print("Loading tweets...")
    features, labels = utils.load_datasets(
        config.CRISIS_TRAIN_DIR, config.NORMAL_TRAIN_DIR, limit=limit
    )
    print("Clearing tweets...")
    features = preprocess.clear_tweets(features)
    return features, labels


def main():
    np.random.seed(32)

    # x, y = visualizzation()
    # x, y, _ = train._process_linear(x, y, max_features=1000)
    # utils.plot_space(x, y)

    train_x, train_y = preprocessing(limit=30000)

    # print("Training Keras...")
    # model = train.train_keras(
    #     train_x, train_y, config.CRISIS_PRE_TRAINED, epochs=15, hidden_size=256
    # )
    # print("Evaluate Keras...")
    # evaluation.evaluate_keras(model)

    print("Training Naive Bayes...")
    model, tfidf_vec = train.train_bayes(train_x, train_y)
    print("Evaluate Bayes...")
    evaluation.evaluate_sklearn(model, tfidf_vec, kind_model="Naive Bayes")

    print("Training SVM...")
    model, tfidf_vec = train.train_svm(train_x, train_y)
    print("Evaluate SVM...")
    evaluation.evaluate_sklearn(model, tfidf_vec, kind_model="SVM")


if __name__ == "__main__":
    main()
