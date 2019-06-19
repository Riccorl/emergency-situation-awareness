import os
import csv
from collections import defaultdict
import config
import re


def read_dataset(in_path: str):

    out_data = defaultdict(list)
    for subdirs, dirs, files in os.walk(in_path):
        for file in files:
            _, file_ext = os.path.splitext(file)
            if file_ext == ".csv":
                print()
                with open(os.path.join(subdirs, file), 'r', encoding='latin1') as csv_file:
                    reader = csv.reader(csv_file)
                    to_read = next(reader)
                    for row in to_read:
                        print(row)
            break


    return out_data


def clear_word(word: str)-> str:
    """
    Remove from a word every special character 
    (tags,hash-tag, number,url,etc).
    :param word: a string of word
    :return: a string (a word) without
    special character
    """
    return re.sub(config.REGEX,"",str(word.lower()))