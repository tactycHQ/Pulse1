#Pulse I
import os
os.environ['TF_KERAS'] = '1'
from data import twitterProcessor
import tensorflow as tf
from keras_bert import get_custom_objects, Tokenizer
from utils.config import get_config_from_json
import numpy as np
import codecs

class Client:

    def __init__(self,model_path):
        self.config = get_config_from_json('.//config.json')
        self.load_path = model_path
        self.vocab_path = self.config.paths.vocab_path
        self.val_data_dir = self.config.paths.val_data_dir
        self.test_data_dir = self.config.paths.test_data_dir
        self.model = tf.keras.models.load_model(self.load_path, custom_objects=get_custom_objects())
        print("Model loaded succesfully from " + self.load_path)

    # Test on unlabelled data
    def test(self):

        data = np.array([
                "'1 more weeek until Spider-Man Far From Home comes out & we are so fucking excited,,',",
                "'was cheesy as hell, but I am here for it,,',",
                "'we went to see again and we sat near an elderly couple and they were just the cutest! this just proves disney magic is timeless,,',",
                "'I’ve seen it 5 times since it came out hahaha. I’ve voted too many times to count ,,',",
                "'Gonna see Rocketman again tomorrow bitch,,',",
                "'Went to see #Rocketman again tonight. Two hours of joyous escapism. Saturday Night's Alright still my favourite. ,,',",
                "'Just cried watching Rocketman, think I'm having a nervous breakdown.,,',",
                "'Longtime EJ/BT fan and I just got home from watching this epic film. I cried almost the entire time, sometimes for happy memories of my youth in the 70s (and my first EJ experiences), and sometimes for sad, but ending with uplifting. Thank you Elton! Definitely award material!,,',",
                "'seeing rocketman tonight FINALLY,,',",
                "'Watching Chernobyl and it’s a struggle. Amazing accounts but so sad,,',",
                "'first episode into #ChernobylHBO its unreal!,,',"
        ])

        token_dict = {}
        with codecs.open(self.vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        tokenizer = Tokenizer(token_dict)

        indices = []
        for index,line in enumerate(data):
            ids, segments = tokenizer.encode(line.strip(),max_len=128)
            indices.append(ids)

        x_test = [indices,np.zeros_like(indices)]
        predictions = self.model.predict(x_test)

        print(np.array(predictions).argmax(axis=-1))


if __name__ == '__main__':
    model_path = 'h5models/run2.h5'
    predict = Client(model_path)
    predict.test()
    # predict.convert_to_pb()

