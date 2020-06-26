##################################### Before run this code ################################
# Here are the things you need to do before run this codes.
# PUT your tokens into 'config.py' file.
# You need to have following files in your project folder
# config.py
# examples_for_seminar.py
# rest_for_seminar.py
# streaming_for_seminar.py
###########################################################################################

import datetime
import examples_for_seminar as e


#
# REST API 1 : Search tweets containing certain keyword
#

# In this example, we search tweeets containing word 'trump' from yesterday to today.

today = str(datetime.datetime.utcnow().date()) # format should be "YYYY-MM-DD"
yesterday = str(datetime.datetime.utcnow().date() - datetime.timedelta(days=1)) # format should be "YYYY-MM-DD"

# Set up parameters
q = {'since': yesterday, 'until': today, 'result_type': 'recent', 'lang': 'en'}

# This function search tweets and store this into json file named 'search.json'.
e.save_search_to_file('', **q)


#
# REST API 2: Collecting tweets of a certain user.
#
# In this example, we collect tweets published by @realDonaldTrump.

# This function store tweets into JSON file, named 'realDonaldTrump-tweets.json'.(#3200)
e.save_user_archive_to_file('')



#
# STREAMING API
#

# In this example, we track 'trump' (and obama)

# this function just print streaming result.
e.track_keywords('') # when you want to stop, press [ctrl + c]
e.track_keywords(['','']) # tweets containing either 'election' or 'obama'

# this function store result into json file named 'keywords_example.json'
e.save_track_keywords('')


