from sklearn.model_selection import train_test_split

import config
import evaluation
import preprocess
import train
import utils


def preprocessing():
    print("Loading tweets...")
    features, labels = utils.load_datasets(
        config.CRISIS_TRAIN_DIR, config.NORMAL_TRAIN_DIR
    )
    print("Clearing tweets...")
    features = preprocess.clear_tweets(features)
    return train_test_split(features, labels, test_size=0.10)


def main():
    train_x, dev_x, train_y, dev_y = preprocessing()

    print("Training Keras...")
    model = train.train_keras(train_x, dev_x, train_y, dev_y, config.CRISIS_PRE_TRAINED)
    evaluation.evaluate_keras(model)

    print("Training Naive Bayes...")
    model, tfidf_vec = train.train_bayes(train_x, train_y)
    evaluation.evaluate_sklearn(model, tfidf_vec, kind_model="Naive Bayes")

    print("Training Svm...")
    model, tfidf_vec = train.train_svm(train_x, train_y)
    evaluation.evaluate_sklearn(model, tfidf_vec, kind_model="Svm")


if __name__ == "__main__":
    main()
