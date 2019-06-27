#Pulse I
import os
os.environ['TF_KERAS'] = '1'
from Dataloader import twitterProcessor
import tensorflow as tf
from keras_bert import get_custom_objects, Tokenizer
from utils.config import get_config_from_json
import numpy as np
import codecs
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

class Client:

    def __init__(self,model_path):
        self.config = get_config_from_json('.//config.json')
        self.load_path = model_path
        self.vocab_path = self.config.paths.vocab_path
        self.val_data_dir = self.config.paths.val_data_dir
        self.test_data_dir = self.config.paths.test_data_dir

    # Flattens data
    def flat_queries(selfs, tweet_path):
        clean_tweets_df = pd.read_pickle(tweet_path)
        title = []
        rawTweet = []
        cleanTweet = []

        for index, row in clean_tweets_df.iterrows():
            for raw, clean in zip(row['rawTweetResults'],row['cleanTweetResults']):
                cleanTweet.append(clean)
                rawTweet.append(raw)
                title.append(row['title'])

        queries_df = pd.DataFrame({'title':title,'rawTweet':rawTweet,'cleanTweet':cleanTweet})
        return queries_df

    def getBERTScore(self,queries_df):

        tweets = queries_df['cleanTweet']

        token_dict = {}
        with codecs.open(self.vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        tokenizer = Tokenizer(token_dict)

        indices = []
        for index,line in enumerate(tweets):
            ids, segments = tokenizer.encode(line.strip(),max_len=128)
            indices.append(ids)

        x_test = [indices,np.zeros_like(indices)]
        predictions = self.model.predict(x_test)

        return predictions

    def writePredictions(self,queries_df,predictions):
        queries_df['bertNEG'] = predictions[:,0]
        queries_df['bertPOS'] = predictions[:, 1]
        queries_df.to_csv("outputs//query_results_bert.csv")
        return queries_df

    def loadBERT(self):
        self.model = tf.keras.models.load_model(self.load_path, custom_objects=get_custom_objects())
        print("Model loaded succesfully from " + self.load_path)


if __name__ == '__main__':
    model_path = 'h5models/run3.h5'
    tweet_path = 'C://Users//anubhav//Desktop//Projects//Gemini1//Database//query_results.pkl'

    client = Client(model_path)
    flatQueries = client.flat_queries(tweet_path)
    client.loadBERT()
    BERTpreds = client.getBERTScore(flatQueries)
    client.writePredictions(flatQueries,BERTpreds)



