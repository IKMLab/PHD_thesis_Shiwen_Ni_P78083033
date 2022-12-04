#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Shiwen Ni"
# Date: 2022/8/8
import os
os.environ['TF_KERAS'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy
from tqdm import tqdm
import numpy as np
from bert4keras.backend import keras, search_layer, K
from sklearn import metrics
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.losses import kullback_leibler_divergence as kld
from utils import *
from hyper_parameters import *
from bert4keras.optimizers import Adam
import copy


class train_data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, pattern_list, *args, **kwargs):
        super(train_data_generator, self).__init__(*args, **kwargs)
        self.pattern_list = pattern_list

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_target_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            for i, pattern in enumerate(self.pattern_list):
                token_ids, segment_ids = tokenizer.encode(first_text=pattern, second_text=text, maxlen=maxlen)
                target_label = 0.0 if i == label else 1.0
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_target_labels.append([target_label])
            if len(batch_token_ids) == self.batch_size * len(self.pattern_list) or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_target_labels = sequence_padding(batch_target_labels)
                yield [batch_token_ids, batch_segment_ids], batch_target_labels
                batch_token_ids, batch_segment_ids, batch_target_labels = [], [], []


class test_data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, pattern="", *args, **kwargs):
        super(test_data_generator, self).__init__(*args, **kwargs)
        self.pattern = pattern

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(first_text=self.pattern, second_text=text, maxlen=maxlen)
            source_ids, target_ids = token_ids[:], token_ids[:]
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids, batch_output_ids], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []

def evaluate(data_generator_list, data, note=""):
    print("\n*******************Start to few-Shot predict on 【{}】*******************".format(note), flush=True)
    patterns_logits = [[] for _ in patterns]
    for i in range(len(data_generator_list)):
        print("\nPattern{}".format(i), flush=True)
        data_generator = data_generator_list[i]
        counter = 0
        for (x, _) in tqdm(data_generator):
            # print(x,"@@@@@@@@@@@@@@@@@@@@2")
            outputs = discriminator.predict(x[:2])
            for out in outputs:
                logit_pos = out.T   # [CLS]+Pattern+sentence 
                patterns_logits[i].append(logit_pos)
                counter += 1

    # Evaluate the results
    trues = [d[1] for d in data]
    preds = []
    for i in range(len(patterns_logits[0])):
        pred = numpy.argmin([logits[i] for logits in patterns_logits])  # min
        preds.append(int(pred))

    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
    print("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
    acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
    return acc


if __name__ == "__main__":

    # Load the hyper-parameters-----------------------------------------------------------
    maxlen = 128  # The max length 128 is used in our paper
    batch_size = 8  # real batch_size = batch_size x num_classes   base = 4  
    num_classes = 2

    # Choose a dataset----------------------------------------------------------------------
    # dataset_names = ['SST-2', 'SST-5', 'MR', 'CR', 'Subj', 'TREC']
    dataset_name = 'TREC'
    lw_loc = 6 if dataset_name == 'TREC' else 3   # Determine the position of the label word

    # Choose a model----------------------------------------------------------------------
    # ['electra-base', 'electra-large']
    model_name = 'electra-base'

    # Load model and dataset class
    bert_model = Model(model_name=model_name)
    dataset = Datasets(dataset_name=dataset_name)

    # Choose a template [0, 1, 2]--------------------------------------------------------
    patterns = dataset.patterns[1]


    # Load the train/dev/test dataset
    # global dev_data, test_data, dev_generator_list, test_generator_list
    train_data = dataset.load_data(dataset.train_path, sample_num=-1, is_shuffle=True)
    train_generator = train_data_generator(pattern_list=patterns, data=train_data,
                                           batch_size=batch_size)

    dev_data = dataset.load_data(dataset.dev_path, sample_num=-1, is_shuffle=True)
    dev_generator_list = []
    for p in patterns:
        dev_generator_list.append(
            test_data_generator(pattern=p, data=dev_data, batch_size=16))

    test_data = copy.deepcopy(dataset.load_data(dataset.test_path, sample_num=-1, is_shuffle=True))
    test_generator_list = []
    for p in patterns:
        test_generator_list.append(
            test_data_generator(pattern=p, data=test_data, batch_size=40))

    
    def rtd_loss(y_true, y_pred):
        # Using categorical_crossentropy to realize softmax
        y_true_ = K.reshape(y_true, (-1, num_classes))
        y_pred_ = K.reshape(y_pred, (-1, num_classes))
        
        loss = K.binary_crossentropy(y_true_, y_pred_, from_logits=False)
        return K.mean(loss)

    # Build ELECTRA model---------------------------------------------------------------------
    tokenizer = Tokenizer(bert_model.dict_path, do_lower_case=True)
    # Load ELECTRA model with RTD head
    electra = build_transformer_model(
        config_path=bert_model.config_path,
        checkpoint_path=bert_model.checkpoint_path,
        model='electra', 
        with_discriminator=True,
        # dropout_rate=0.1,
        return_keras_model=False
    )
    rtd_output = electra.model.output
    rtd_output = keras.layers.Lambda(lambda x: x[:, lw_loc])(rtd_output)  # 这个就是预测值
    discriminator = keras.models.Model(electra.model.inputs, rtd_output)

    class Evaluator(keras.callbacks.Callback):
        def __init__(self):
            self.best_val_acc = 0.
            self.best_epoch = 1
        def on_epoch_end(self, epoch, logs=None):
            val_acc = evaluate(dev_generator_list, dev_data, note="Dev Set")
            test_acc = evaluate(test_generator_list, test_data, note="Test Set")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
            print(
                u'val_acc: %.5f, test_acc: %.5f, best_val_acc: %.5f, best_epoch: %.0f' %
                (val_acc, test_acc, self.best_val_acc, self.best_epoch)
            )
            print('------------------'*10)
            print('\n')
            print('Start the next epoch traning:')
    evaluator = Evaluator()

    print(model_name, discriminator.summary())

    def adversarial_training(model, embedding_name, epsilon=0.01, alpha=0.1, beta=0.1):
        """Add adversarial training to the model
         This function is to be used after compile().
        """
        if model.train_function is None:  # If no train_function
            model._make_train_function()  # make train_function
        old_train_function = model.train_function  # Backup old train_function

        # Find the Embedding layer
        for output in model.outputs:
            embedding_layer = search_layer(output, embedding_name)
            if embedding_layer is not None:
                break
        if embedding_layer is None:
            raise Exception('Embedding layer not found')

        # Calculate the Embedding gradient
        embeddings = embedding_layer.embeddings  # Embedding矩阵
        gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
        gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

        # 封装为函数
        inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
        )  # 所有输入层
        embedding_gradients = K.function(
            inputs=inputs,
            outputs=[gradients],
            name='embedding_gradients',
        )  # 封装为函数

        def train_function(inputs):  # 重新定义训练函数
            loss1 = old_train_function(inputs)
            grads = embedding_gradients(inputs)[0]  # Embedding梯度
            delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
            p = model.predict(inputs)
            K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
            loss_adv = old_train_function(inputs)  # 有扰动后的输出loss    
            q = model.predict(inputs)
            loss_kl = K.mean(kld(p,q))
            K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
            return loss1 + [alpha*loss_adv[0]] + [beta*loss_kl]
            # return [beta*loss_kl]    # unsupervised
        model.train_function = train_function  # 覆盖原训练函数

    discriminator.compile(loss=rtd_loss, optimizer=Adam(1e-5)) # 1e-5

    # 写好函数后，启用对抗训练只需要一行代码
    # adversarial_training(discriminator, 'Embedding-Token', epsilon=0.5, alpha=0.2, beta=0.1)
 
    # Few-Shot traning, predict and evaluate-------------------------------------------------------
    print('Start traning:')
    discriminator.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator],
        shuffle=True
    )

    # zero-shot
    val_acc = evaluate(test_generator_list, test_data, note="Test Set")
    print(val_acc)
