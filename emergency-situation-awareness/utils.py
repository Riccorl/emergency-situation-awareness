import csv
import os
import re
import zipfile
from typing import List, Dict, Tuple

import nltk
import numpy as np
from nltk.corpus import stopwords

import config


def read_dataset(filename: str) -> List[str]:
    """
    Read the dataset line by line.
    :param filename: file to read
    :return: a list of lines
    """
    with open(filename, encoding="utf8") as file:
        f = (line.strip() for line in file)
        return [line for line in f if line]


def write_dataset(filename: str, lines: List[str]):
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
        for k, *v in dictionary.items():
            file.write(k + "\t" + "\t".join(v[0]) + "\n")


def merge_txt_files(input_file: List[str], output_filename: str):
    """
    Merge the given text files.
    :param input_file: list of strings.
    :param output_filename: filename of the output file
    """
    with open(output_filename, "w", encoding="utf8") as out_file:
        out_file.writelines(line + "\n" for line in input_file)


def load_datasets() -> Tuple[List[str], List[int]]:
    """
    This method is used to handle all datasets path to read.
    :return: a list of tweets
    """
    # parse crisis tweets
    crisis_tweets = read_crisisnlp() + read_crisilex()
    crisis_tweets_label = [1] * len(crisis_tweets)
    print("Number of crisis tweets:", len(crisis_tweets))

    # parse non-crisis tweets
    normal_tweets = read_normal()
    normal_tweets_label = [0] * len(normal_tweets)
    print("Number of non-crisis tweets:", len(normal_tweets))
    return crisis_tweets + normal_tweets, crisis_tweets_label + normal_tweets_label


def read_crisisnlp() -> List[str]:
    """
    Read crisis tweets from crisisnlp folder.
    :return: list of tweets.
    """
    tweets = []
    for file in config.CRISISNLP_DIR.glob("./*.csv"):
        tweets += _read_csv(file)
    return tweets


def read_crisilex() -> List[str]:
    """
    Read crisis tweets from crisislex folder.
    :return: list of tweets.
    """
    tweets = []
    for file in config.CRISISLEX_DIR.glob("./*.csv"):
        tweets += _read_crisislex_csv(file)
    return tweets


def read_normal() -> List[str]:
    """
    Read non-crisis tweets from normal folder.
    :return: list of tweets.
    """
    tweets = []
    for file in list(config.NORMAL_DIR.glob("./*.csv"))[:6]:
        tweets += _read_csv(file)
    return tweets


def _read_csv(filename) -> List[str]:
    """
    Extract tweet from csv file.
    :param filename: csv file.
    :return: a list of tweets.
    """
    with open(filename, encoding="latin1") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        return [row[-1] for row in csv_reader]


def _read_crisislex_csv(filename) -> List[str]:
    """
    Extract tweet from csv file.
    :param filename: csv file.
    :return: a list of tweets.
    """
    with open(filename, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        return [row[1] for row in csv_reader]


def split_dataset(filename: str, n_split: int):
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
            data.append(" ".join(row))

    batch = len(data) // n_split
    for i in range(0, len(data), batch):
        j = i + batch
        filename_batch = str(filename).split(".")[0] + "_" + str(n_split) + ".csv"
        print("Writing", filename_batch)
        write_dataset(filename_batch, data[i:j])
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
        if w2v.vectors_norm:
            vec_norm = w2v.vectors_norm[i]
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


def clean_embeddings(path_input: str, path_output: str, size: int):
    """
    Clean embeddings by removing non lemma_synset vectors.
    :param path_input: path to original embeddings.
    :param path_output: path to cleaned embeddings.
    :return:
    """
    old_emb = read_dataset(path_input)
    filtered = [vector for vector in old_emb if "_bn:" in vector]
    write_dataset(path_output, [str(len(filtered)) + " " + str(size)] + filtered)



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
