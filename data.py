import os
import numpy as np
from tqdm import tqdm
from keras_bert import Tokenizer
import codecs
import pandas as pd

class twitterProcessor():

    def __init__(self,vocab_path, data_dir, SEQ_LEN):
        self.vocab_path = vocab_path
        self.data_dir = data_dir
        self.seq_len = SEQ_LEN

    def get_train_examples(self,data_dir):
        token_dict = {}
        with codecs.open(self.vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        self.tokenizer = Tokenizer(token_dict)

        with open(data_dir,'r',encoding='utf-8') as f:
                reader=f.readlines()
                x_train, y_train = self.create_examples(reader,"train")
        return x_train, y_train

    def create_examples(self,lines,set_type):
        examples=[]
        indices, labels = [], []
        for index,line in enumerate(lines):
            guid = "%s-%s" % (set_type, index)
            split_line=line.strip().split('+++$+++')
            ids, segments = self.tokenizer.encode(split_line[1],max_len=self.seq_len)
            sentiment= split_line[0]
            indices.append(ids)
            labels.append(sentiment)
        return [indices,np.zeros_like(indices)],np.array(labels)


    def get_test_examples(self,data_dir):
        token_dict = {}
        with codecs.open(self.vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        self.tokenizer = Tokenizer(token_dict)

        with open(data_dir,'r',encoding='utf-8') as f:
                reader=f.readlines()
                x_test = self.create_test_examples(reader,"train")
                return x_test

    def create_test_examples(self,lines,set_type):
        examples=[]
        indices = []
        for index,line in enumerate(lines):
            guid = "%s-%s" % (set_type, index)
            ids, segments = self.tokenizer.encode(line.strip(),max_len=self.seq_len)
            indices.append(ids)
        return [indices,np.zeros_like(indices)]
