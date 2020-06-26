#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Those are revised code of [examples.py] written by Copyright (C) 2016 Pascal Jürgens and Andreas Jungherr (See License.txt).
# Revised by T.Kim.

# The code revision details:
# - Add time setting for Europe/Berlin
# - Add argument: print_user_archive(user)
# - Add argument and change file name setting: save_user_archive_to_file(user)
# - Add argument: save_user_archive_to_database(user)
# - Add argument: track_keywords(k_list)
# - Add argument: save_track_keywords(k_list)
# - change MST to LT var in all functions so that we can set arbitrary timezone
# - Add new function save_search_to_file(query,**kwargs)


"""
Examples for accessing the API
------------------------------
These are some examples demonstrating the use of provided functions for
gathering data from the Twitter API.

Requirements:
    - depends on API access modules rest.py and streaming.py
"""

import rest_for_seminar as rest
import streaming_for_seminar as streaming
import logging
import json



#
# Helper Functions
#


def print_tweet(tweet):
    """
    Print a tweet as one line:
    user: tweet
    """
    logging.warning(
        u"{0}: {1}".format(tweet["user"]["screen_name"], tweet["text"]))


def print_notice(notice):
    """
    This just prints the raw response, such as:
    {u'track': 1, u'timestamp_ms': u'1446089368786'}}
    """
    logging.error(u"{0}".format(notice))

#
# Examples for extracting Tweets
#




def print_user_archive(user):
    """
    Fetch all available tweets for one user and print them, line by line
    :param user: str indicating screen_name(without @) or user_id
    """
    archive_generator = rest.fetch_user_archive(user)
    for page in archive_generator:
        for tweet in page:
            print_tweet(tweet)


def save_user_archive_to_file(user):
    """
    Fetch all available tweets for one user and save them to a text file, one tweet per line.
    :param user: str indicating screen_name(without @) or user_id
    :return: a json file, named user-tweets.json, contains recent 3200 tweets published from the user.
    """
    filename = str(user).__add__('-tweets.json')
    with open(filename, "w") as f:
        archive_generator = rest.fetch_user_archive(user)
        for page in archive_generator:
            for tweet in page:
                f.write(json.dumps(tweet) + "\n")
    logging.warning(u"Wrote tweets from the user")


def track_keywords(k_list):
    """
    Track two keywords with a tracking stream and print machting tweets and notices.
    To stop the stream, press ctrl-c or kill the python process.
    :param: k_list: list of keywords
    """
    keywords = k_list
    stream = streaming.stream(
        on_tweet=print_tweet, on_notification=print_notice, track=keywords)


def save_track_keywords(k_list):
    """
    Track two keywords with a tracking stream and save machting tweets.
    To stop the stream, press ctrl-c or kill the python process.
    :param: k_list : list of keywords.
    """
    # Set up file to write to
    outfile = open("keywords_example.json", "w")

    def save_tweet(tweet):
        json.dump(tweet, outfile)
        # Insert a newline after one tweet
        outfile.write("\n")
    keywords = k_list
    try:
        stream = streaming.stream(
            on_tweet=save_tweet, on_notification=print_notice, track=keywords)
    except (KeyboardInterrupt, SystemExit):
        logging.error("User stopped program, exiting!")
        outfile.flush()
        outfile.close()



def save_search_to_file(query,**kwargs):
    """
    Save search results using keywords (max period: last 7 days)
    :param query: query for search
    :param kwargs: parameter including 'since' and 'until', which indicate the period (utc)

    example:
    q = {'since': '2018-11-24','until':'2018-11-25', 'result_type': 'recent', 'lang':'de'}
    save_search_to_file('CDU', **q)

    if you want to use local time,

    start_date = LT.localize(datetime.datetime(2018,11,1))
    start_date = str(utc.normalize(start_date).date())

    set start_date as 'since' value and do the same for 'until'

    """
    with open("search.json","w") as f:
        generator = rest.search_tweets_period(query,**kwargs)
        for page in generator:
            for tweet in page:
                f.write(json.dumps(tweet)+"\n")
    logging.warning(u"Wrote tweets from search keyword to json file.")







