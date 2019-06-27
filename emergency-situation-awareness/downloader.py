import argparse
import csv
import logging

from tqdm import tqdm
from twython import Twython
import tweepy


def tweet_dowloader(filename, path_outputh):
    logging.basicConfig(
        format="%(levelname)s - %(asctime)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    consumer_key = "YOUR_KEY"
    consumer_secret = "YOUR_KEY"
    access_token = "YOUR_KEY"
    access_token_secret = "YOUR_KEY"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    tweets_ids = _read_tweet_ids(filename)
    lookup_tweets(tweets_ids, api, path_outputh)
    return


def lookup_tweets(tweet_ids, api, path_outputh):
    tweet_count = len(tweet_ids)
    with open(path_outputh, mode="w", encoding="utf8") as out_file:
        try:
            for i in range((tweet_count // 100) + 1):
                # Catch the last group if it is less than 100 tweets
                end_loc = min((i + 1) * 100, tweet_count)
                full_tweets = api.statuses_lookup(id_=tweet_ids[i * 100: end_loc], tweet_mode='extended')
                # for j in range(i * 100, end_loc):
                out_file.writelines(_fast_clean_tweet(t) + "\n" for t in full_tweets if t)
        except tweepy.TweepError:
            print("Something went wrong, quitting...")


def _read_tweet_ids(filename):
    with open(filename, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        return [row[0].replace("'", "") for row in csv_reader]


def _fast_clean_tweet(tweet):
    if tweet.retweeted_status:
        print("it's retweet")
        tweet_text = tweet.retweeted_status.full_text
    else:
        print("it's not retweet")
        tweet_text = tweet.full_text
    tweet_text = tweet_text.strip()
    tweet_text = tweet_text.replace('\n', '')
    return tweet_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        help="input file",
        dest="input",
    )
    parser.add_argument(
        help="output file",
        dest="output",
    )
    return parser.parse_args()


if __name__ == '__main__':
    # args = parse_args()
    tweet_dowloader("../data/ids/2014-08-california_earthquake_2014_20140824_vol-1.json.csv", "../data/downloaded/et_california_2.txt")
