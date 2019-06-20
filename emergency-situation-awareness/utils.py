import codecs
import csv
import os
import re
import zipfile
import nltk
from collections import defaultdict
import nltk
from nltk.corpus import stopwords



def read_datasets(paths: list) -> defaultdict:
    """
    This method is used to handle all datasets path to read.
    :param paths: paths to read
    :return: a dictionary of dictionaries
    """
    out_data = defaultdict()
    for path in paths:
        out_data.update(read_dataset(path))

    return out_data


def read_dataset(path: str) -> defaultdict:
    """
    This method is used to read a dataset handling it extension.
    :param path: path to read
    :return: a dictionary of dictionaries
    """
    out_data = defaultdict()

    for subdirs, dirs, files in os.walk(path):
        for file in files:
            _, file_ext = os.path.splitext(file)

            if file.endswith(".csv"):

                # this one is to run the code. For me we need to remove MACOSX folder
                if not subdirs.endswith("__MACOSX"):
                    out_data.update(read_csv(os.path.join(subdirs, file)))

    return out_data


def read_csv(filepath: str) -> defaultdict:
    """
    Csv reader for the used datasets.
    :param filepath: file to read
    :return: a dictionary of dictionaries having
                tweet_id -> {text, class}
    """

    out_data = defaultdict()

    reader = csv.reader(codecs.open(filepath, "r", "latin1"))
    next(reader)
    for row in reader:
        if all([piece is not None for piece in [row[0], row[7], row[9]]]):
            out_data[row[0].replace("'", "")] = {
                "text": row[7],
                "class": row[9],
            }

    return out_data


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

def stop_words(all_language:bool =False) -> list:
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
    nltk.download('stopwords',quiet=True)

    if all_language:
        return stopwords.words(stopwords.fileids())
    else:
        return stopwords.words("english")