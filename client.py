#Pulse I
import os
os.environ['TF_KERAS'] = '1'
from Dataloader import twitterProcessor
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
                "'I am so excited for Ozarks",
                "I have seen many shows better than Ozark",
                "Ozark is strictly ok",
                "Can't wait for Ozark",
                "I am not happy",
                "The action in Ozark isn't really all that great"
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
        np.savetxt('.//outputs//predictions.csv',np.array(predictions),delimiter=",")
        print(np.array(predictions))


if __name__ == '__main__':
    model_path = 'h5models/run2.h5'
    predict = Client(model_path)
    predict.test()
    # predict.convert_to_pb()

