import argparse
import csv
from typing import List

import tweepy


def tweet_dowloader(path_input: str, path_outputh: str):
    """
    Download tweets from a list of ids.
    :param path_input: ids in input.
    :param path_outputh: where to save the tweets.
    :return:
    """
    consumer_key = "bBU4XnEMpraeHwZlK50zAhxK5"
    consumer_secret = "bOaeDo1HsVphtMnq9xq2tYEAPU28xzju5ss1OM1YicbUmgIU5F"
    access_token = "224185852-90ENd3QMyKwYujvi32LgJW2qNHoN7fem8S8SXUYS"
    access_token_secret = "sV8sdardQuy0mce7EGBb6BYUv6URcw8A3xuocjHk2lZyj"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    tweets_ids = _read_tweet_ids(path_input)
    lookup_tweets(tweets_ids, api, path_outputh)
    return


def lookup_tweets(tweet_ids: List, api: tweepy.API, path_outputh: str):
    """
    Retrieve tweets from a list of ids.
    :param tweet_ids: tweet ids.
    :param api: Tweepy API object.
    :param path_outputh: where to save the tweets.
    :return:
    """
    tweet_count = len(tweet_ids)
    with open(path_outputh, mode="w", encoding="utf8") as out_file:
        for i in range((tweet_count // 100) + 1):
            end_loc = min((i + 1) * 100, tweet_count)
            try:
                full_tweets = api.statuses_lookup(
                    id_=tweet_ids[i * 100 : end_loc], tweet_mode="extended"
                )
                out_file.writelines(
                    _fast_clean_tweet(t) + "\n" for t in full_tweets if t
                )
            except tweepy.TweepError as e:
                print("Something went wrong, error:", e)


def _read_tweet_ids(path_input: str):
    """
    Read a file containing tweets id (crisis NLP tweets) and returns them as a list.
    :param path_input: file containing ids.
    :return: a list of tweet ids.
    """
    with open(path_input, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        return [row[0].replace("'", "") for row in csv_reader]


def _fast_clean_tweet(tweet):
    """
    Remove new line char within text.
    :param tweet: Tweet in input.
    :return: text of the tweet.
    """
    tweet_text = _extract_text(tweet)
    tweet_text = tweet_text.strip()
    tweet_text = tweet_text.replace("\n", "")
    return tweet_text


def _extract_text(tweet):
    """
    Check if tweet is RT and returns the extended text of the tweet.
    :param tweet: Tweet in input.
    :return: non-cutted tweet text.
    """
    if hasattr(tweet, "retweeted_status"):
        try:
            tweet_text = tweet.retweeted_status.extended_tweet.full_text
        except AttributeError:
            tweet_text = tweet.retweeted_status.full_text
    else:
        try:
            tweet_text = tweet.extended_tweet.full_text
        except AttributeError:
            tweet_text = tweet.full_text
    return tweet_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(help="input file", dest="input")
    parser.add_argument(help="output file", dest="output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tweet_dowloader(args.input, args.output)
