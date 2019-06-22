from tensorflow import keras
from keras_bert import AdamWarmup, calc_train_steps, get_custom_objects, load_trained_model_from_checkpoint, backend


class BertModel():

    def __init__(self,config_path, checkpoint_path,seq_len,batch_size,epochs, lr):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.load_pre_trained_model()

    def load_pre_trained_model(self):
        self.pretrained_model = load_trained_model_from_checkpoint(
                                                self.config_path,
                                                self.checkpoint_path,
                                                training=True,
                                                trainable=True,
                                                seq_len = self.seq_len,
        )

    def finetune_model(self,data_size,loss_fn,metrics):
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
            loss=loss_fn,
            metrics=[metrics],
        )
        self.model = model
        return self.model

    def load(self, checkpoint_path):
        """
        loads an H5 file
        :param checkpoint_path:file path
        :return:
        """
        self.model = keras.models.load_model(checkpoint_path,custom_objects = get_custom_objects())



