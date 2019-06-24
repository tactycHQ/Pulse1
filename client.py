#Pulse I
import os
import numpy as np
os.environ['TF_KERAS'] = '1'
from data import twitterProcessor
import tensorflow as tf
from model import BertModel
from keras_bert import bert, gen_batch_inputs, get_base_dict, get_custom_objects
from tensorflow.python.ops.math_ops import erf, sqrt
from utils.config import get_config_from_json


def main():
    # Processing config file
    config = get_config_from_json('.//config.json')
    load_flag = config.setup.load
    data_dir = './/data//predict2.csv'

    processor = twitterProcessor(config.paths.vocab_path, data_dir, config.model.seq_len)
    x_test = processor.get_test_examples(data_dir)

    model = BertModel(config.paths.config_path,
                      config.paths.ckpt_path,
                      config.model.seq_len,
                      config.model.batch_size,
                      config.model.epochs,
                      config.model.lr)

    model.load(".\\h5models\\"+config.setup.run_num)
    print("Model loaded succesfully from "+config.setup.run_num)

    predictions = model.model.predict(x_test)
    print(predictions)

#Gelu function as per GitHub bug for TF2
def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))

if __name__ == '__main__':
    bert.gelu = gelu
    main()

