import csv
import os
import re
import string
import zipfile
from typing import List, Set, Dict

from nltk.corpus import stopwords


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


def read_datasets(crisis_paths, normal_paths):
    """
    This method is used to handle all datasets path to read.
    :param paths: paths to read
    :return: a list of tweets
    """
    # preprocessing stuffs
    stops = set(stopwords.words("english")) | set(string.punctuation)

    # parse crisis tweets
    crisis_tweets = _read_tweets(crisis_paths, stops)
    crisis_tweets_label = [1] * len(crisis_tweets)
    print("Number of crisis tweets:", len(crisis_tweets))

    # parse non-crisis tweets
    normal_tweets = _read_tweets(normal_paths, stops)
    normal_tweets_label = [0] * len(normal_tweets)
    print("Number of non-crisis tweets:", len(normal_tweets))
    return crisis_tweets + normal_tweets, crisis_tweets_label + normal_tweets_label


def _read_tweets(filenames: List, stops: Set = None) -> List:
    """
    Extract tweet from multiple csv files.
    :param filenames: list of csv file.
    :param stops: set of stop words and punctuation to remove.
    :return: a list of cleaned tweets.
    """
    tweets = []
    for filename in filenames:
        with open(filename, encoding="latin1") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            next(csv_reader)
            for row in csv_reader:
                if stops:
                    tweet = _clear_tweet(row[-1].lower().split(), stops)
                else:
                    tweet = row[-1].lower().split()
                if len(tweet) > 2:
                    tweets.append(tweet)
    return tweets


def _clear_tweet(tweet: str, stops: Set) -> List:
    """
    Clean the tweet in input.
    :param tweet: tweet to clean.
    :param stops: set of stop words and punctuation to remove.
    :return: tweet cleaned.
    """
    return [word for word in tweet if word not in stops]


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


def clear_word(text: str) -> str:
    """
    Remove from a string of text every special
    character (tags,hash-tag, number,url,etc).
    :param text: a string of text
    :return: a string (a text) without
    special character
    """

    return " ".join((" ".join(re.compile("[^a-zA-Z\d\s:]").split(text))).split())


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
