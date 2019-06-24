#Pulse I
import os
os.environ['TF_KERAS'] = '1'
from data import twitterProcessor
import tensorflow as tf
from keras_bert import get_custom_objects
from utils.config import get_config_from_json
import numpy as np

class Predict:

    def __init__(self,model_path):
        self.config = get_config_from_json('.//config.json')
        self.load_path = model_path
        self.val_data_dir = self.config.paths.val_data_dir
        self.test_data_dir = self.config.paths.test_data_dir
        self.model = tf.keras.models.load_model(self.load_path, custom_objects=get_custom_objects())
        print("Model loaded succesfully from " + self.load_path)

    # Test on unlabelled data
    def test(self):
        processor = twitterProcessor(self.config.paths.vocab_path, self.test_data_dir, self.config.model.seq_len)
        x_test = processor.get_test_examples(self.test_data_dir)

        predictions = self.model.predict(x_test)
        np.savetxt(".//outputs//predictions.csv",predictions,delimiter=",")
        print("Predictions saved")

    #On validation data
    def validate(self):
        # Validation on dev set
        processor = twitterProcessor(self.config.paths.vocab_path, self.val_data_dir, self.config.model.seq_len)
        x_val, y_val = processor.get_train_examples(self.val_data_dir)
        results = self.model.evaluate(x_val,y_val,batch_size=self.config.model.batch_size)
        print('test loss, test acc:', results)

    def convert_to_pb(self):
        tf.contrib.saved_model.save_keras_model(self.model, ".\\served_models\\1", custom_objects=get_custom_objects())
        print("Model saved succesfully as PB ")

if __name__ == '__main__':
    model_path = 'h5models/run2.h5'
    predict = Predict(model_path)

    # predict.test()
    predict.validate()
    # predict.convert_to_pb()

