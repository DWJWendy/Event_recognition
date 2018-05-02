#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-25 下午7:03
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : other_model.py
"""

from Program.Preprocess import datapreprocess
from sklearn.metrics.classification import f1_score, recall_score, precision_score
import numpy as np
import pandas as pd
import logging
from sklearn.svm import NuSVC
from sklearn.naive_bayes import MultinomialNB
from ast import literal_eval
from Program.Model.model import do_format
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


class Contrast_Classifer(object):
    def __init__(self,  target_train_path, test_x_path, test_y_path):
        self.svm_model = NuSVC()
        self.bayes_model = MultinomialNB()
        self.nn_model = MLPClassifier()
        self.target_data = datapreprocess.read_file(target_train_path)
        self.test_x = datapreprocess.read_file(test_x_path)
        self.test_y = datapreprocess.read_file(test_y_path)

    @staticmethod
    def do_fit(train_x, train_y, test_x, model):
        model.fit(train_x, train_y)
        predict_y = model.predict(test_x)
        return predict_y

    def unified_structure(self, item_data):
        unified_result = []
        for i in range(len(item_data.index)):
            unified_result.append(item_data[i])
        return unified_result

    def convert_one_hot(self, item_data):
        target_words = self.target_data["Split_content"]
        all_words = set()
        for i in range(len(target_words.index)):
            all_words = all_words | set(literal_eval(target_words.ix[i, "Split_content"]))
        all_words = list(all_words)
        onehot_result = pd.DataFrame(np.zeros((len(item_data.index), len(all_words))),
                                     columns=all_words, index=item_data.index)
        for i in range(len(item_data.index)):
            words = literal_eval(item_data.ix[i, "Split_content"])
            for word in words:
                if word in all_words:
                    onehot_result.ix[i, word] = onehot_result.ix[i, word] + 1
                else:
                    pass
        return onehot_result


    @staticmethod
    def metrics(y_true, y_predict):
        logger.info("计算分类指标...")
        F_value = f1_score(y_true, y_predict, average="weighted")
        Recall_value = recall_score(y_true, y_predict, average="weighted")
        Precision_value = precision_score(y_true, y_predict, average="weighted")
        return F_value, Recall_value, Precision_value

    def run(self):
        logger.info("计算svm+word2vec的效果...")
        train_x = self.unified_structure(self.target_data["Sentence_vec"].apply(lambda x: do_format(x)[0]))
        train_y = np.array(self.target_data["Label"])
        test_x = self.unified_structure(self.test_x.copy()["Sentence_vec"].apply(lambda x: do_format(x)[0]))
        test_y = np.array(self.test_y.copy()["Label"])
        predict_y = self.do_fit(train_x, train_y, test_x, self.svm_model)
        print(test_y)
        print(predict_y)
        F_value, Recall_value, Precision_value = self.metrics(y_true=test_y, y_predict=predict_y)
        print("SVM+Word2vec结果：" + str(F_value) + "\t" + str(Recall_value) + "\t" + str(Precision_value))

        logger.info("计算SVM+bag of words的效果...")
        train_x = self.convert_one_hot(item_data=self.target_data["Split_content"])
        test_x = self.convert_one_hot(item_data=self.test_x["Split_content"])
        predict_y = self.do_fit(train_x, train_y, test_x, self.svm_model)
        print(test_y)
        print(predict_y)
        F_value, Recall_value, Precision_value = self.metrics(y_true=test_y, y_predict=predict_y)
        print("SVM+bag of word结果：" + str(F_value) + "\t" + str(Recall_value) + "\t" + str(Precision_value))

        logger.info("计算bayes+bag of words的效果...")
        predict_y = self.do_fit(train_x, train_y, test_x, self.bayes_model)
        print(test_y)
        print(predict_y)
        F_value, Recall_value, Precision_value = self.metrics(y_true=test_y, y_predict=predict_y)
        print("bayes结果：" + str(F_value) + "\t" + str(Recall_value) + "\t" + str(Precision_value))

        logger.info("计算多层感知器效果...")
        predict_y = self.do_fit(train_x, train_y, test_x, self.nn_model)
        print(test_y)
        print(predict_y)
        F_value, Recall_value, Precision_value = self.metrics(y_true=test_y, y_predict=predict_y)
        print("nn结果：" + str(F_value) + "\t" + str(Recall_value) + "\t" + str(Precision_value))

if __name__ == "__main__":
    # 对比实验, 识别事件模型
    target_train_path = "../Data_2/pre_process/5_150_1_target_represent.csv"
    test_x_path = "../Data_2/pre_process/test_data.csv"
    test_y_path = "../Data_2/pre_process/test_data_label.csv"
    contrast_model = Contrast_Classifer(target_train_path, test_x_path, test_y_path)
    contrast_model.run()
