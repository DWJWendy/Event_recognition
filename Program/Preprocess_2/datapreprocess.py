#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-28 上午9:10
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : datapreprocess.py
"""
# !/usr/bin/env python
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
import random
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
    def __init__(self, raw_path, target_path, stop_words_path):
        """
        :type raw_path, label_path, stop_words_path: str
        """
        # 读取原始数据
        self.process_path = "../Data_2/pre_process/"
        self.raw_data = read_file(path=raw_path)
        # 读取标签数据,并将标签进行映射
        self.target_data = read_file(path=target_path)
        self.stopwords = self.read_stopwords_file(path=stop_words_path)
        # 对语料库和标签数据都进行分词
        # self.corpora_split_content = self.split_sentence(
        #     item_data=self.raw_data, filename="corpora_split_content.csv", field=["Weibo_id", "Content"])
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
        contents["Split_content"] = contents["Content"].apply(lambda x: self.jieba_word(item=str(x)))
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
        model = Word2Vec. \
            load(self.process_path + 'window_' + str(window) + '_size_' + str(size) + '_sg_' + str(sg) + '.vec.model')
        split_contents["Sentence_vec"] = \
            split_contents["Split_content"].apply(lambda x: self.word_represent(item_data=literal_eval(x), model=model))
        write_file(self.process_path + filename, split_contents)
        return split_contents

    @staticmethod
    def form_test_data(item, k=600):
        indexs = list(item.index)
        indexs = random.sample(indexs, k=k)
        test_data = item.ix[indexs, ]
        write_file(path="../Data_2/raw/test_data.csv", item_data=test_data)
        return test_data

    def run(self, window=5, size=150, sg=1):
        # # 读取分词后的语料库数据以及标签数据
        corpora_split_content = read_file(self.process_path + "corpora_split_content.csv")
        target_split_content = read_file(self.process_path + "target_split_content.csv")
        #
        # # 训练词向量,对语料库和标签数据都表征
        self.train_word2vec(corpora_split_content, window=window, size=size, sg=sg, data_from=True)
        # 语料库表征
        filename = str(window) + "_" + str(size) + "_" + str(sg) + "_sentence_represent.csv"
        corpora_data = self.sentence_represent(corpora_split_content, filename=filename,
                                               field=["Weibo_id", "Split_content", "Content"], window=window, size=size, sg=sg)
        # 目标表征
        filename = str(window) + "_" + str(size) + "_" + str(sg) + "_target_represent.csv"
        self.sentence_represent(target_split_content, filename=filename,
                                field=["Weibo_id", "Split_content", "Label"], window=window, size=size, sg=sg)

        # 测试集数据表征
        self.form_test_data(corpora_data, k=600)
        logger.info("数据预处理完成...")


if __name__ == "__main__":
    raw_path = "../Data_2/raw/clean_company_corpora.csv"
    target_path = "../Data_2/raw/target.csv"
    stopwords_path = "../Data_2/raw/Stopwords.txt"
    window, size, sg = 5, 150, 1
    test = DataProcess(raw_path=raw_path, target_path=target_path, stop_words_path=stopwords_path)
    test.run()
