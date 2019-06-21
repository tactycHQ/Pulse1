import os
import numpy as np
os.environ['TF_KERAS'] = '1'
from data import twitterProcessor
from model import BertModel
from keras_bert import bert, gen_batch_inputs, get_base_dict
from tensorflow.python.ops.math_ops import erf, sqrt
from tensorflow.data import Dataset


def main():

    pretrained_path = 'uncased_L-12_H-768_A-12/'
    config_path = pretrained_path + 'bert_config.json'
    checkpoint_path = pretrained_path + 'bert_model.ckpt'
    vocab_path = pretrained_path + 'vocab.txt'
    data_dir = "data/train.csv"

    SEQ_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 1
    LR = 1e-4

    processor = twitterProcessor(vocab_path, data_dir, SEQ_LEN)
    x_train, y_train = processor.get_train_examples(data_dir)

    model = BertModel(config_path,
                      checkpoint_path,
                      SEQ_LEN,
                      BATCH_SIZE,
                      EPOCHS,
                      LR)
    model.load_pre_trained_model()
    model = model.create_model(y_train.shape[0])
    model.fit(x_train,
              y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE
              )
    model.save('.//h5models//run1.h5')

def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))

if __name__ == '__main__':
    bert.gelu = gelu
    main()

