from tensorflow import keras
from keras_bert import AdamWarmup, calc_train_steps
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import backend



class BertModel():

    def __init__(self,config_path, checkpoint_path,seq_len,batch_size,epochs, lr):
        tf_keras = 1
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model = None


    def load_pre_trained_model(self):
        self.pretrained_model = load_trained_model_from_checkpoint(
                                                self.config_path,
                                                self.checkpoint_path,
                                                training=True,
                                                trainable=True,
                                                seq_len = self.seq_len,
        )

    def create_model(self,data_size):
        inputs = self.pretrained_model.inputs[:2]
        dense = self.pretrained_model.get_layer('NSP-Dense').output
        outputs = keras.layers.Dense(units=2, activation='softmax')(dense)

        decay_steps, warmup_steps = calc_train_steps(data_size,
                                                     batch_size=self.batch_size,
                                                     epochs=self.epochs,
        )

        model = keras.models.Model(inputs, outputs)
        model.compile(
            AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=self.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
        )
        self.model = model
        return self.model


