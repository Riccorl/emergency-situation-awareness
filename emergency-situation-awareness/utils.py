import csv
import os
import re
import zipfile
from typing import List, Dict, Tuple

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import learning_curve


def read_txt(filename: str) -> List[str]:
    """
    Read the dataset line by line.
    :param filename: file to read
    :return: a list of lines
    """
    with open(filename, encoding="utf8") as file:
        f = (line.strip() for line in file)
        return [line for line in f if line]


def write_txt(filename: str, lines: List[str]):
    """
    Writes a list of string in a file.
    :param filename: path where to save the file.
    :param lines: list of strings to serilize.
    :return:
    """
    with open(filename, "w", encoding="utf8") as file:
        file.writelines(line + "\n" for line in lines)


def read_dictionary(filename: str) -> Dict:
    """
    Open a dictionary from file, in the format key -> value
    :param filename: file to read.
    :return: a dictionary.
    """
    with open(filename) as file:
        return {k: v for k, *v in (l.split() for l in file)}


def write_dictionary(filename: str, dictionary: Dict):
    """
    Writes a dictionary as a file.
    :param filename: file where to save the dictionary.
    :param dictionary: dictionary to serialize.
    :return:
    """
    with open(filename, mode="w") as file:
        for k, v in dictionary.items():
            file.write(k + " " + str(v) + "\n")


def merge_txt_files(input_file: List[str], output_filename: str):
    """
    Merge the given text files.
    :param input_file: list of strings.
    :param output_filename: filename of the output file
    """
    with open(output_filename, "w", encoding="utf8") as out_file:
        out_file.writelines(line + "\n" for line in input_file)


def load_datasets(
    crisis_path, normal_path, limit: int = 30000
) -> Tuple[List[str], List[int]]:
    """
    This method is used to handle all datasets path to read.
    :return: a list of tweets
    """
    # parse crisis tweets
    crisis_tweets = read_datasets(crisis_path, limit)
    crisis_tweets_label = [1] * len(crisis_tweets)
    print("Number of crisis tweets:", len(crisis_tweets))

    # parse non-crisis tweets
    normal_tweets = read_datasets(normal_path, limit)
    normal_tweets_label = [0] * len(normal_tweets)
    print("Number of non-crisis tweets:", len(normal_tweets))
    return crisis_tweets + normal_tweets, crisis_tweets_label + normal_tweets_label


def read_datasets(path, limit: int = 30000) -> List[str]:
    """
    Read crisis tweets from crisisnlp folder.
    :return: list of tweets.
    """
    tweets = []
    for file in path.glob("./*.txt"):
        tweets += read_txt(file)[:limit]
    return tweets


def split_csv(filename: str, n_split: int):
    """
    Split a large text file in smaller files.
    :param filename: file to split.
    :param n_split: number of parts to split.
    :return:
    """
    data = []
    with open(filename, encoding="latin1") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            data.append(row[-1])

    batch = len(data) // n_split
    for i in range(0, len(data), batch):
        j = i + batch
        filename_batch = str(filename).split(".")[0] + "_" + str(n_split) + ".txt"
        print("Writing", filename_batch)
        write_txt(filename_batch, data[i:j])
        n_split -= 1


def split_txt(filename: str, n_split: int):
    """
    Split a large text file in smaller files.
    :param filename: file to split.
    :param n_split: number of parts to split.
    :return:
    """
    data = read_txt(filename)
    batch = len(data) // n_split
    for i in range(0, len(data), batch):
        j = i + batch
        filename_batch = str(filename).split(".")[0] + "_" + str(n_split) + ".txt"
        print("Writing", filename_batch)
        write_txt(filename_batch, data[i:j])
        n_split -= 1


def unzip_all(paths: list):
    """
    This method is used to unzip all files.
    :param paths: a list of path to unzip
    :return: None
    """
    for path in paths:
        for subdirs, dirs, files in os.walk(path):
            for file in files:
                _, file_ext = os.path.splitext(file)
                if file_ext == ".zip":
                    unzip(subdirs, file)


def unzip(main_folder: str, file: str):
    """
    This method is used to unzip a file
    :param main_folder: folder to extract
    :param file: file to unzip
    :return: None
    """
    zip_ref = zipfile.ZipFile(os.path.join(main_folder, file), "r")
    zip_ref.extractall(main_folder)
    zip_ref.close()


def clear_text(text: str) -> str:
    """
    This method remove from a string of text
    every special character (tags,hash-tag, number,
    url,etc).
    :param text: a string of text
    :return: a string (a text) without
    special character
    """

    return " ".join((" ".join(re.compile("[^a-zA-Z\d\s:]").split(text))).split())


def stop_words(all_language: bool = False) -> list:
    """
    This method is used to generate stop words.
    The default language is English but setting
    the boolean variable to true it generates the
    stop words for all language.
    :param all_language: a boolean variable used
    as flag for the languages.
    :return: a list containing the stop
    words.
    """
    nltk.download("stopwords", quiet=True)

    if all_language:
        return stopwords.words(stopwords.fileids())
    else:
        return stopwords.words("english")


def restrict_w2v(w2v, restricted_word_set):
    """
    Retrain from w2v model only words in the restricted word set.
    :param w2v:
    :param restricted_word_set:
    :return:
    """
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        vec_norm = w2v.vectors_norm[i] if w2v.vectors_norm else []
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            if vec_norm:
                new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)
    if new_vectors_norm:
        w2v.vectors_norm = np.array(new_vectors_norm)
    return w2v


def w2v_txt_to_bin(path_input: str, path_output: str):
    """

    :param path_input:
    :param path_output:
    :return:
    """
    w2v = gensim.models.KeyedVectors.load_word2vec_format(path_input, binary=False)
    w2v.save_word2vec_format(path_output, binary=True)


def flatten(input_list: List[List[str]]) -> List[str]:
    """
    This method is used to flat a list
    :param input_list: list to flat
    :return: flatted list
    """
    return [" ".join(elem) for elem in input_list]


def _plot_learning_curve(
    estimator,
    title,
    x,
    y,
    scoring,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    x : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    fig = plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    fig.savefig("plot.png")
    return plt


def plot_space(x, y):
    """
    Plot data in 2d space.
    :param x: features.
    :param y: labels.
    :return:
    """
    data_2d = TruncatedSVD(n_components=2, n_iter=200, random_state=42).fit_transform(x)
    plt.grid()
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=y)
    plt.show()


def plot_keras(history):
    """
    Plot validation and training accuracy, validation and training loss over time.
    """
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(history.history["loss"])
    axes[0].plot(history.history["val_loss"])
    axes[0].legend(
        ["loss", "val_loss"],
        loc="upper right",
        frameon=True,
        facecolor="white",
        fontsize="large",
    )

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(history.history["acc"])
    axes[1].plot(history.history["val_acc"])
    axes[1].legend(
        ["acc", "val_acc"],
        loc="lower right",
        frameon=True,
        facecolor="white",
        fontsize="large",
    )

    plt.savefig("keras.png")
    plt.show()


def plot_learning_curve(x, y, estimator, cv, scoring):
    """
    Plot the learning curve.
    :param x: features.
    :param y: labels.
    :param estimator:
    :param cv:
    :param scoring:
    :return:
    """
    title = "Learning Curves"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    _plot_learning_curve(
        estimator,
        title,
        x,
        y,
        scoring,
        ylim=(0.7, 1.01),
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.01, 1.0, 10),
    )
    plt.show()


def timer(start: float, end: float) -> str:
    """
    Timer function. Compute execution time from strart to end (end - start).
    :param start: start time
    :param end: end time
    :return: end - start
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
