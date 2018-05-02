#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-23 上午9:55
@Author  : 毛毛虫_wendy
@E-mail  : dengwenjun818@gmail.com
@blog    : mmcwendy.info
@File    : model.py
"""

from Program.Preprocess import datapreprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.classification import f1_score, recall_score, precision_score
import numpy as np
import pandas as pd
import logging
from ast import literal_eval
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


def do_format(content):
    return np.array(list(filter(lambda x: x, content.replace("[", "").
                                replace("]", "").replace("\n", "").split(" ")))).astype(float).reshape(1, -1)


class Trigger(object):
    def __init__(self, train_path, corpora_split_path, model_path):
        self.trigger_path = "../Data/trigger/"
        self.model_path = model_path
        self.target_train_data = datapreprocess.read_file(train_path)
        self.corpora_split_content = datapreprocess.read_file(corpora_split_path)
        self.seed_words, self.top_seed_trigger, self.seed_event_vec, self.event_words, self.event_vec = \
            None, None, None, None, None

    def form_seed_words(self):
        """
        :return: 得到最初的触发器词
        """
        logger.info("形成种子事件词...")
        target_train_data = self.target_train_data.copy()[["Label", "Split_content"]]
        trigger_words = {}
        for i in range(len(target_train_data.index)):
            if int(target_train_data.ix[i, "Label"]) not in trigger_words:
                trigger_words[int(target_train_data.ix[i, "Label"])] = literal_eval(
                    target_train_data.ix[i, "Split_content"])
            else:
                trigger_words[int(target_train_data.ix[i, "Label"])]. \
                    extend(literal_eval(target_train_data.ix[i, "Split_content"]))
        datapreprocess.write_json_file(path=self.trigger_path + "init_trigger.json", item_data=trigger_words)
        return trigger_words

    @staticmethod
    def sort_words(items, top=400):
        """
        根据频数对其排序,选取前top个数据
        :param items:type[word1,word2,...,]
        :param top:前top个高频词
        :return:
        """
        word_frequency = {}
        for item in items:
            if item not in word_frequency:
                word_frequency[item] = 1
            else:
                word_frequency[item] += 1
        if len(items) >= top:
            sort_trigger_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:top]
        else:
            sort_trigger_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
        return sort_trigger_words

    def seed_trigger(self, seed_words):
        """
        :return: 每一类事件触发对应的words,且根据频数对其排序,选取前top个数据
        """
        top_seed_trigger = {}
        for event, value in seed_words.items():
            top_seed_trigger[event] = self.sort_words(items=value)
        datapreprocess.write_file(path=self.trigger_path + "top_event_trigger.csv",
                                  item_data=pd.DataFrame(top_seed_trigger).T)
        datapreprocess.write_json_file(path=self.trigger_path + "top_event_trigger.json", item_data=top_seed_trigger)
        return top_seed_trigger

    @staticmethod
    def compute_vec(item, model, weight=False, method=1):
        """
        计算每一个事件的加权向量,加权方式分为两种
        :param item: eg [["\u53d1\u5c55", 88], ["\u8fdb\u884c", 77], ["\u6253\u9020", 73]] 对应词和频数
        :param model: 加载Word2vec模型
        :return:
        """

        e_words = pd.DataFrame(item, columns=["word", "frequency"]).copy()
        e_words["vec"] = e_words["word"].apply(lambda x: model.wv[x] if x in model else np.zeros(150))
        if weight:
            if method == 1:
                # 对权重的处理-1
                frequency_min = e_words["frequency"].min()
                frequency_max = e_words["frequency"].max()
                e_words["weight"] = e_words["frequency"]. \
                    apply(lambda x: (x - frequency_min) / (frequency_max - frequency_min))
            if method == 2:
                # 对权重的处理-2
                frequency_sum = e_words["frequency"].sum()
                e_words["weight"] = e_words["frequency"].apply(lambda x: x / frequency_sum)
            # 加权向量
            e_words["weight_vec"] = e_words["weight"] * e_words["vec"]
            event_vec = sum(e_words["weight_vec"]) / len(e_words.index)
            return event_vec
        else:  # 不加权向量
            event_vec = sum(e_words["vec"]) / len(e_words.index)
            return event_vec

    def seed_vec(self, e_trigger, filename, weight=False, method=1):
        """
        :return: 得到每类事件的向量
        """
        logger.info("计算种子数据中的事件向量...")
        model = Word2Vec.load(self.model_path)
        seed_vec = {}
        for e, value in e_trigger.items():
            seed_vec[e] = self.compute_vec(item=value, model=model, weight=weight, method=method)
        datapreprocess.write_file(path=self.trigger_path + filename, item_data=pd.DataFrame(seed_vec).T)
        return seed_vec

    def add_new_word(self, threshold=0.8, top=2000, enrich=True, source_from=1, new=False):
        """
        从测试集中提取top=1000的新词，加入到event
        :return:
        """
        if enrich:
            logger.info("丰富种子动词形成触发器词...")
            # 加载语料库中高频新词
            if source_from == 1:
                corpora_split_content = self.corpora_split_content.copy()
            else:
                # 加载测试集中高频新词
                corpora_split_content = \
                    datapreprocess.read_file(path="../Data/pre_process/target_test_x.csv")["Split_content"]
            model = Word2Vec.load(self.model_path)
            new_trigger_words, seed_event_vec, all_word = self.top_seed_trigger.copy(), self.seed_event_vec, []
            final_trigger_words = {}
            # sorted the corpora high frequency words
            for i in range(len(corpora_split_content.index)):
                all_word.extend(literal_eval(corpora_split_content.ix[i, "Split_content"]))
            high_frequency_words = self.sort_words(all_word, top=top)
            new_trigger_words_list = []
            for value in new_trigger_words.values():
                for wd in value:
                    new_trigger_words_list.append(wd[0])
            # add new word to trigger.
            if not new:
                for item in high_frequency_words:
                    if item[0] not in new_trigger_words_list:
                        cosine_value = []
                        for i in sorted(seed_event_vec.keys()):
                            if item[0] in model:
                                vec_pair = [seed_event_vec[i], model.wv[item[0]]]
                                cosine_value.append(cosine_similarity(vec_pair)[1, 0])
                        if cosine_value and max(cosine_value) >= threshold:
                            event = cosine_value.index(max(cosine_value)) + 1
                            logger.info("增加词：{word},频率:{frequency},相似度:{simi},对应事件{event}".
                                        format(word=item[0], frequency=item[1], simi=max(cosine_value), event=event))
                            new_trigger_words[event].append(item)
            else:
                new_trigger_words = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
                for item in high_frequency_words:
                    cosine_value = []
                    for i in sorted(seed_event_vec.keys()):
                        if item[0] in model:
                            vec_pair = [seed_event_vec[i], model.wv[item[0]]]
                            cosine_value.append(cosine_similarity(vec_pair)[1, 0])
                    if cosine_value and max(cosine_value) >= threshold:
                        event = cosine_value.index(max(cosine_value)) + 1
                        logger.info("增加词：{word},频率:{frequency},相似度:{simi},对应事件{event}".
                                    format(word=item[0], frequency=item[1], simi=max(cosine_value), event=event))
                        new_trigger_words[event].append(item)

            datapreprocess.write_json_file(path=self.trigger_path + "event_words_add_new.json", item_data= new_trigger_words)
            high_top = min(list(map(lambda x: len(x), new_trigger_words.values())))
            logger.info("每类标签的词数{num}".format(num=high_top))
            for key, value in new_trigger_words.items():
                final_trigger_words[key] = value[:high_top]
            datapreprocess.write_json_file(path=self.trigger_path + "final_event_words.json", item_data=final_trigger_words)
            datapreprocess.write_file(path=self.trigger_path + "final_event_word.csv",
                                      item_data=pd.DataFrame(final_trigger_words).T)
            return final_trigger_words
        else:
            return self.top_seed_trigger.copy()

    def compute_event_vec(self, weight_2, method):
        logger.info("计算最终的事件向量..")
        event_trigger_words = self.event_words
        e_vec = self.seed_vec(event_trigger_words, filename="event_vec.csv", weight=weight_2, method=method)
        return e_vec

    def run(self, weight_1=True, weight_2=True, method=1, threshold=0.7, top=2000, enrich=True, source_from=2, new=False):
        logger.info("计算种子词...")
        self.seed_words = self.form_seed_words()
        logger.info("计算前top个高频种子词")
        self.top_seed_trigger = self.seed_trigger(self.seed_words)
        # 计算种子向量
        filename = "seed_vec_"+str(weight_1)+"_"+str(method)+".csv"
        logger.info("计算计算种子向量,权重：{weight},方法：{method}".format(weight=weight_1, method=method))
        self.seed_event_vec = self.seed_vec(self.top_seed_trigger, filename, weight=weight_1, method=method)
        # 丰富种子词形成触发器
        self.event_words = self.add_new_word(threshold=threshold, top=top, enrich=enrich,
                                             source_from=source_from, new=new)
        # 计算触发器向量
        self.event_vec = self.compute_event_vec(weight_2=weight_2, method=method)
        return self.event_vec


class Trigger_Classifer(Trigger):
    def __init__(self, train_path, corpora_split_path, model_path, tain_x_path, train_y_path, test_x_path, test_y_path,
                 weight_1, weight_2, method, threshold, top, enrich, source_from, new):
        Trigger.__init__(self, train_path, corpora_split_path, model_path)
        self.event_vec = Trigger.run(self, weight_1=weight_1, weight_2=weight_2, method=method, threshold=threshold,
                                     top=top, enrich=enrich, source_from=source_from, new=new)
        self.model_path = "../Data/model/"
        self.train_x = datapreprocess.read_file(tain_x_path)
        self.train_y = datapreprocess.read_file(train_y_path)
        self.test_x = datapreprocess.read_file(test_x_path)
        self.test_y = datapreprocess.read_file(test_y_path)

    def compute_event_cosine(self, item_x, event_vec, shreshold=0.7):
        """
        计算每条微博与每类事件的相似度,选择相似度最大,如果相似度最大且对于阈值,则加入对应的企业事件,否则为null事事件
        :param item_x: np.array类型, 每一条微博的向量
        :param event_vec: 每一个事件的中心向量
        :param shreshold: 阈值（默认0.7）
        :return:
        """
        cosine_simi = []
        for e in sorted(event_vec.keys()):
            cosine_simi.append(cosine_similarity(do_format(item_x), event_vec[e].reshape(1, -1))[0][0])
        max_value = max(cosine_simi)
        if max_value >= shreshold:
            return cosine_simi.index(max_value) + 1
        else:
            return 6

    def validation(self, event_vec):
        logger.info("计算测试集对应的企业事件...")
        test_x = self.test_x.copy()
        predict_result = pd.DataFrame(index=test_x.index, columns=["predict_label"])
        predict_result["predict_label"] = test_x["Sentence_vec"]. \
            apply(lambda x: self.compute_event_cosine(x, event_vec))
        return predict_result

    @staticmethod
    def fenzi(x, y, event):
        fz = 0
        for i in range(len(x)):
            if x[i] == y[i] == event:
                fz += 1
        return fz

    def metrics(self, y_true, y_predict):
        y_true, y_predict = list(y_true), list(y_predict)
        event = [1, 2, 3, 4, 5, 6]
        event_precision, event_recall, event_fvalue, average = {}, {}, {}, {}
        for i in range(len(y_true)):
            for j in event:
                recall_fm = len(list(filter(lambda x: x == j, y_true)))
                precison_fm = len(list(filter(lambda x: x == j, y_predict)))
                fz = self.fenzi(x=y_true, y=y_predict, event=j)
                event_recall[j] = fz / recall_fm
                event_precision[j] = fz / precison_fm
                event_fvalue[j] = (2 * event_recall[j] * event_precision[j]) / (event_recall[j]+ event_precision[j])
        average["precision"] = sum(event_precision.values()) / 6
        average["recall"] = sum(event_recall.values()) / 6
        average["f_value"] = sum(event_fvalue.values()) / 6
        return event_precision, event_recall, event_fvalue, average

    @staticmethod
    def metrics_2(y_true, y_predict ):
        logger.info("计算分类指标...")
        F_value = f1_score(y_true, y_predict, average="weighted")
        Recall_value = recall_score(y_true, y_predict, average="weighted")
        Precision_value = precision_score(y_true, y_predict, average="weighted")
        return F_value, Recall_value, Precision_value

    def run(self, **kwargs):
        y_predict = np.array(self.validation(self.event_vec)["predict_label"])
        y_true = np.array(self.test_y["Label"])
        F_value, Recall_value, Precision_value = self.metrics_2(y_true, y_predict)
        print("Our model结果：" + str(F_value) + "\t" + str(Recall_value) + "\t" + str(Precision_value))
        return self.metrics(y_true, y_predict)

