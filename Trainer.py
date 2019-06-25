#Pulse I
import os
import numpy as np
os.environ['TF_KERAS'] = '1'
from Dataloader import twitterProcessor
import tensorflow as tf
from Model import BertModel
from keras_bert import bert, gen_batch_inputs, get_base_dict, get_custom_objects
from utils.config import get_config_from_json
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

class Trainer:

    def __init__(self):
        self.config = get_config_from_json('.//config.json')
        self.callbacks=[]
        self.init_callbacks()
        self.train()


    def train(self):
        train_data_dir = self.config.paths.train_data_dir
        val_data_dir = self.config.paths.val_data_dir

        train_processor = twitterProcessor(self.config.paths.vocab_path, train_data_dir, self.config.model.seq_len)
        x_train, y_train = train_processor.get_train_examples(train_data_dir)
        num_samples = y_train.shape[0]
        batch_size = self.config.model.batch_size

        val_processor = twitterProcessor(self.config.paths.vocab_path, val_data_dir, self.config.model.seq_len)
        x_val, y_val = train_processor.get_train_examples(val_data_dir)

        model = BertModel(self.config.paths.config_path,
                          self.config.paths.ckpt_path,
                          self.config.model.seq_len,
                          self.config.model.batch_size,
                          self.config.model.epochs,
                          self.config.model.lr)
        print("Model instantiated")

        model = model.compile_model(num_samples,
                                    self.config.model.loss_fn,
                                    self.config.model.metrics)
        print("Model compiled. Commencing training")

        model.fit(x=x_train,
                  y=y_train,
                  validation_data=(x_val,y_val),
                  epochs=self.config.model.epochs,
                  batch_size=batch_size,
                  callbacks=self.callbacks
                  )

        model.save(self.config.callbacks.save_dir+self.config.setup.run_num)
        print("Model saved succesfully as H5 file")

    def init_callbacks(self):
        self.callbacks.append(
            CSVLogger('.//logs//training_log.csv',
                      separator=',',
                      append=False)
        )

        self.callbacks.append(
            TensorBoard(
                log_dir='.//logs//tensorboard_logs//',
                write_graph=True,
            )
        )
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.config.callbacks.save_dir+self.config.setup.run_num,
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

if __name__ == '__main__':
    trainer = Trainer()

