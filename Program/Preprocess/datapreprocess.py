#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-21 下午4:37
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : datapreprocess.py
"""

import pandas as pd
import logging, json
import jieba.posseg as pseg
from ast import literal_eval
import numpy as np
from sklearn.cross_validation import train_test_split
from numpy import nan as NA
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


def read_file(path: object) -> object:
    logger.info("正在读取文件...")
    return pd.read_csv(path, sep="\t")


def write_file(path, item_data):
    logger.info("正在写入文件...")
    item_data.to_csv(path, sep="\t", encoding="utf-8")


def write_json_file(path, item_data):
    logger.info("正在写入json文件...")
    with open(path, "w", encoding="utf-8") as dump_f:
        json.dump(item_data, dump_f)


def read_json_file(path):
    logger.info("正在读取json文件...")
    with open(path, "r", encoding="utf-8") as dump_f:
        return json.load(dump_f)


class DataProcess(object):
    def __init__(self, raw_path, label_path, stop_words_path):
        """
        :type raw_path, label_path, stop_words_path: str
        """
        self.process_path = "../Data/pre_process/"
        self.label_num = {"招募": 1, "合作": 2, "研发": 3, "推广": 4, "销售": 5, "无": 6}
        # 读取原始数据
        self.raw_data = read_file(path=raw_path)
        logger.info("原始语料库数据条数：{num}".format(num=len(self.raw_data.index)))
        # 将原始数据中空博文的删掉,并以weibo_id为index
        self.clean_data = self.delete_na(item=self.raw_data)
        # 读取标签数据,并将标签进行映射
        self.target_data = self.convert_label(item=read_file(path=label_path).dropna())
        logger.info("带标签的目标数据条数：{num}".format(num=len(self.target_data.index)))
        self.stopwords = self.read_stopwords_file(path=stop_words_path)
        self.train_x, self.test_x, self.train_y, self.test_y = None, None, None, None
        # 对语料库和标签数据都进行分词
        # self.corpora_split_content = self.split_sentence(
        #     item_data=self.clean_data, filename="corpora_split_content.csv", field=["Weibo_id", "Content"])
        # self.target_split_content = self.split_sentence(
        #     item_data=self.target_data, filename="target_split_content.csv", field=["Weibo_id", "Content", "Label"])

    @staticmethod
    def read_stopwords_file(path):
        """
        :param path: the path of the stopwords file
        :return:
        """
        logger.info('读取停用词文件，返回停用词列表...')
        with open(path, 'r', encoding='utf-8') as f:
            stopwords = f.readlines()
        return map(lambda x: x.replace('\n', ''), stopwords)

    def delete_na(self, item):
        """
        方法：去掉内容内空的实例
        :return:输入原始数据 self.raw_data, 返回内容都不为空的新数据
        """
        logger.info("对原始语料库进行缺失值处理...")
        clean_data = item[item["Content"].notnull()]
        logger.info("删除语料库空微博后的数据条数：{num}".format(num=len(clean_data.index)))
        write_file(self.process_path + "corpora_clean_data.csv", clean_data)
        return clean_data

    def convert_label(self, item):
        """
        :param item: 将对应的label标签进行映射
        :return:
        """
        logger.info("对事件标签进行映射...")
        item["Label"] = item["Label"].apply(lambda x: self.label_num[x])
        write_file(self.process_path + "target_data.csv", item)
        return item

    def jieba_word(self, item, allowPOS=("v", "vn"), CX=False):
        """
        :param item: 需要分词的内容，使用结巴进行分词，并去除一字词和停用词等
        :param allowPOS: allowPOS=('v','vn')表示所需词性(默认)
        :param CX: CX=True表示返回带有词性
        :return:split_content(list) or NA
        """
        split_content = []
        words = pseg.cut(sentence=item)
        for word, flag in words:
            if allowPOS and flag in allowPOS and word not in self.stopwords and len(word) > 1:
                split_content.append((word + '_' + flag)) if CX else split_content.append(word)
            elif allowPOS == () and word not in self.stopwords and len(word) > 1:
                split_content.append((word + '_' + flag)) if CX else split_content.append(word)
        if split_content:
            return split_content
        else:
            return NA

    def split_sentence(self, item_data, filename, field):
        logger.info("正在结巴分词...")
        contents = item_data[field]
        contents["Split_content"] = contents["Content"].apply(lambda x: self.jieba_word(item=x))
        logger.info("分词后的语料库数据条数：{num}".format(num=len(contents.index)))
        contents = contents.dropna(axis=0)
        logger.info("分词后的语料库删除空词序列的数据条数：{num}".format(num=len(contents.index)))
        write_file(self.process_path + filename, item_data=contents)
        return contents

    def train_word2vec(self, contents, window, size, sg, data_from=True):
        """
        :param contents: 待训练的已分词的文本数据list=[[],[],[]]
        :param window: 模型窗口,默认为5
        :param size: 词向量维度，默认为150
        :param sg: 模型选择默认为skip-gram
        :return:
        """
        train_contents = []
        if data_from:  # 如果contents来自于数据读取
            for i in range(len(contents.index)):
                train_contents.append(literal_eval(contents.ix[i, "Split_content"]))
        else:
            train_contents = np.array(contents["Split_content"])
        logger.info("将分词后的数据进行模型训练...")
        model = Word2Vec(sentences=train_contents, window=window, size=size, sg=sg, min_count=1, workers=4)
        model.save(self.process_path + 'window_' + str(window) + '_size_' + str(size) + '_sg_' + str(sg) + '.vec.model')
        logger.info("词向量训练完成...")
        return model

    @staticmethod
    def word_represent(item_data, model, size=150):
        item_vec = np.zeros(size)
        length = len(item_data)
        for word in item_data:
            if word in model:
                item_vec = item_vec + model.wv[word]
        item_vec = item_vec / length
        return item_vec

    def sentence_represent(self, split_contents, filename, field, window, size, sg):
        logger.info("正在用word2vec表征句子...")
        split_contents = split_contents[field]
        model = Word2Vec.\
            load(self.process_path + 'window_' + str(window) + '_size_' + str(size) + '_sg_' + str(sg) + '.vec.model')
        split_contents["Sentence_vec"] = \
            split_contents["Split_content"].apply(lambda x: self.word_represent(item_data=literal_eval(x), model=model))
        write_file(self.process_path + filename, split_contents)
        return split_contents

    def split_target(self, x_column, y_column):
        target_represent = read_file(self.process_path + "target_represent.csv")
        item_x = target_represent[x_column]
        item_y = target_represent[y_column]
        train_x, test_x, train_y, test_y = train_test_split(item_x, item_y, test_size=0.2, random_state=0)
        write_file(self.process_path + "target_train_x.csv", train_x)
        write_file(self.process_path + "target_test_x.csv", test_x)
        write_file(self.process_path + "target_train_y.csv", train_y)
        write_file(self.process_path + "target_test_y.csv", test_y)
        return train_x, test_x, train_y, test_y

    def run(self, window=5, size=150, sg=1):
        # 读取分词后的语料库数据以及标签数据
        corpora_split_content = read_file(self.process_path + "corpora_split_content.csv")
        target_split_content = read_file(self.process_path + "target_split_content.csv")

        # 训练词向量,对语料库和标签数据都表征
        self.train_word2vec(corpora_split_content, window=window, size=size, sg=sg, data_from=True)
        # 语料库表征
        self.sentence_represent(corpora_split_content, filename="sentence_represent.csv",
                                field=["Weibo_id", "Split_content"], window=window, size=size, sg=sg)
        # 数据标签表征
        self.sentence_represent(target_split_content, filename="target_represent.csv",
                                field=["Weibo_id", "Split_content", "Label"], window=window, size=size, sg=sg)
        self.train_x, self.test_x, self.train_y, self.test_y = self.split_target(
            x_column=["Label", "Sentence_vec", "Split_content"], y_column=["Label"])
        logger.info("数据预处理完成...")
