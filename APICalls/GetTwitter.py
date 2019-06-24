import tweepy as tw
import numpy as np
from dotmap import DotMap
import pandas as pd
import json

consumer_key='TCAQaSJq0qAjsWyKZdakGssNN'
consumer_secret='o55EyHnqW5NNA05ds29Nvpmg7VkTkpdY2s76EwA6oUfIv8siea'
access_token_key='3722095873-ums9qpIH3g7Y3YJ5kvh3nNMbbslg1gvXrY1Tq8K'
access_token_secret='yhwDe9kOlU0boWNMlsXYxS6CVfkJvJfLF2Y8NoRn5PlXQ'

auth = tw.AppAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token_key, access_token_secret)

api = tw.API(auth)
limit = api.rate_limit_status()
# limit_dict = json.load(limit)
limit = DotMap(limit)
print(limit.resources.search)

class GetTwitter():

    def __init__(self):
        pass

    def getTweetsbyQuery(self,query,max_tweets,date_since):

        tweet_text = []
        tweet_location = []

        query = query + "-filter:retweets"
        date_since = date_since
        tweets = tw.Cursor(api.search,
                  q=query,
                  tweet_mode = 'extended',
                  lang="en",
                  since=date_since).items(max_tweets)
        for tweet in tweets:
            tweet_text.append(tweet.full_text)
            tweet_location.append(tweet.user.location)
        tweet_text = np.array(tweet_text)

        df = pd.DataFrame(tweet_text)
        df.to_csv("tweets.csv")
        tweet_location = np.array(tweet_text)
        return tweet_text, tweet_location

if __name__ == '__main__':
    query = 'avengers'
    max_tweets = 10
    date_since = "2019-06-01"

    getTwitter = GetTwitter()
    tweet_text, tweet_location = getTwitter.getTweetsbyQuery(query,max_tweets,date_since)
    print(tweet_text)