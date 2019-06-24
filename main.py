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
    data_dir = config.paths.data_dir

    processor = twitterProcessor(config.paths.vocab_path, data_dir, config.model.seq_len)
    x_train, y_train = processor.get_train_examples(data_dir)
    num_samples = y_train.shape[0]


    model = BertModel(config.paths.config_path,
                      config.paths.ckpt_path,
                      config.model.seq_len,
                      config.model.batch_size,
                      config.model.epochs,
                      config.model.lr)

    if load_flag == False:
        try:
            model = model.finetune_model(num_samples, config.model.loss_fn,config.model.metrics)
            model.fit(x_train,
                      y_train,
                      epochs=config.model.epochs,
                      batch_size=config.model.batch_size
                      )
            model.save('.//h5models//'+config.setup.run_num)
            print("Model saved succesfully as "+config.setup.run_num)
        except Exception as ex:
            print(ex)
            print("Unable to create new fine tuned BERT model")
    else:
        try:
            model.load(".\\h5models\\"+config.setup.run_num)
            print("Model loaded succesfully from "+config.setup.run_num)
            tf.saved_model.save(model.model,".\\served_models\\1",signatures=None)
            print("Model converted to PB")
        except Exception as ex:
            print(ex)
            print("Unable to load model")

#Gelu function as per GitHub bug for TF2
def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))

if __name__ == '__main__':
    bert.gelu = gelu
    main()

